"""
PyTorch Dataset for SBI over cardiovascular physiology.

Two dataset modes:
- CVDataset: all 28 waveforms (24 continuous z-scored + 4 binary valves),
  25-param theta. Used for the full-channel CNN baseline.
- ReducedCVDataset: 4 pressure waveforms + 5 scalars (Pas mean/max/min, SV,
  HR), 24-param theta (HR moved to observation). Used for the reduced-input CNN.
"""

import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


PARAM_KEYS = [
    "AVD", "Bla", "Blv", "Bra", "Brv",
    "Cas", "Cvp", "Cvs", "Eap",
    "Eedref_la", "Eedref_lv", "Eedref_ra", "Eedref_rv",
    "Emax_LA", "Emax_LV", "Emax_RA", "Emax_RV",
    "HR", "Rap", "Ras", "Tmax", "Tmax_a",
    "Vs", "τ", "τ_a",
]

# HR is observable (measured), so it moves from theta to x in the reduced mode
_HR_IDX = PARAM_KEYS.index("HR")
PARAM_KEYS_INFER = [k for k in PARAM_KEYS if k != "HR"]  # 24-dim

WAVE_KEYS_CONT = [
    "Pap", "Pas", "Pla", "Plv", "Pra", "Prv", "Pvp", "Pvs",
    "Qap", "Qas", "Qla", "Qlv", "Qra", "Qrv", "Qvp", "Qvs",
    "Vap", "Vas", "Vla", "Vlv", "Vra", "Vrv", "Vvp", "Vvs",
]

WAVE_KEYS_VALVE = ["av", "mv", "pv", "tv"]

N_PARAMS = len(PARAM_KEYS)       # 25
N_CHANNELS = len(WAVE_KEYS_CONT) + len(WAVE_KEYS_VALVE)  # 28
T = 201

# Channel group slices within the stacked (N_CHANNELS, T) tensor
_SLICE_P     = slice(0, 8)   # Pap..Pvs  — pressure
_SLICE_Q     = slice(8, 16)  # Qap..Qvs  — flow
_SLICE_V     = slice(16, 24) # Vap..Vvs  — volume
_SLICE_VALVE = slice(24, 28) # av,mv,pv,tv

# 4*8 + 3*8 + 3*8 + 1*4
N_SUMSTATS = 84

# ── Reduced observation constants ─────────────────────────────────────────────
WAVE_KEYS_REDUCED = ["Prv", "Pra", "Pvp", "Pap"]  # 4 pressure waveforms
_PAS_IDX = WAVE_KEYS_CONT.index("Pas")
_VLV_IDX = WAVE_KEYS_CONT.index("Vlv")
N_REDUCED_CHANNELS = len(WAVE_KEYS_REDUCED)        # 4
N_SCALARS          = 5                              # Pas mean/max/min, SV, HR
OBS_DIM            = N_REDUCED_CHANNELS * T + N_SCALARS  # 809
N_CONT             = len(WAVE_KEYS_CONT)            # 24
N_VALVE            = len(WAVE_KEYS_VALVE)           # 4


def compute_summary_stats(x: torch.Tensor) -> torch.Tensor:
    """
    x : (N, N_CHANNELS*T) flat z-scored waveforms
    returns (N, N_SUMSTATS=84) domain-specific summary statistics

    Pressure (8 ch): mean, systolic (max), diastolic (min), pulse pressure
    Flow     (8 ch): mean, peak (max), min
    Volume   (8 ch): EDV (max), ESV (min), stroke volume (max-min)
    Valves   (4 ch): fraction of time open (mean)
    """
    w = x.view(x.shape[0], N_CHANNELS, T)

    p = w[:, _SLICE_P, :]
    p_sys = p.amax(-1)
    p_dia = p.amin(-1)

    q = w[:, _SLICE_Q, :]

    v = w[:, _SLICE_V, :]
    v_ed = v.amax(-1)
    v_es = v.amin(-1)

    valve = w[:, _SLICE_VALVE, :]

    return torch.cat([
        p.mean(-1), p_sys, p_dia, p_sys - p_dia,   # 4*8 = 32
        q.mean(-1), q.amax(-1), q.amin(-1),          # 3*8 = 24
        v_ed, v_es, v_ed - v_es,                     # 3*8 = 24
        valve.mean(-1),                              #   4
    ], dim=-1)


class CVDataset(Dataset):
    def __init__(self, data_dir, index_entries, stats):
        self.data_dir = data_dir
        self.index = index_entries
        self._handles = {}

        w = stats["waves"]
        self.wave_mean = torch.tensor(
            [w[k]["mean"] for k in WAVE_KEYS_CONT], dtype=torch.float32
        ).unsqueeze(1)  # (24, 1) — broadcasts over (24, 201)
        self.wave_std = torch.tensor(
            [w[k]["std"] for k in WAVE_KEYS_CONT], dtype=torch.float32
        ).unsqueeze(1)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        entry = self.index[idx]
        path = os.path.join(self.data_dir, entry["file"])
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
        g = self._handles[path][entry["group"]]

        # theta: raw parameter values, shape (25,)
        theta = torch.tensor(
            [float(g[f"parameters/{k}"][()]) for k in PARAM_KEYS],
            dtype=torch.float32,
        )

        # continuous waveforms: z-scored, shape (24, 201)
        waves_cont = torch.from_numpy(
            np.stack([g[f"waves/{k}"][:] for k in WAVE_KEYS_CONT]).astype(np.float32)
        )
        waves_cont = (waves_cont - self.wave_mean) / (self.wave_std + 1e-8)

        # valve waveforms: binary float as-is, shape (4, 201)
        waves_valve = torch.from_numpy(
            np.stack([g[f"waves/{k}"][:] for k in WAVE_KEYS_VALVE]).astype(np.float32)
        )

        # x: all 28 channels stacked flat → (28*201,) for sbi
        x = torch.cat([waves_cont, waves_valve], dim=0).reshape(-1)

        return theta, x

    def close(self):
        for fh in self._handles.values():
            fh.close()
        self._handles.clear()


class ReducedCVDataset(Dataset):
    """
    Returns (theta_infer, x_reduced) where:
      theta_infer : (24,)  — all params except HR
      x_reduced   : (809,) — 4 z-scored waveforms (4*201) + 5 scalars
                             scalars: Pas mean, Pas max, Pas min, SV, HR_z
    """

    def __init__(self, data_dir, index_entries, stats):
        self.data_dir = data_dir
        self.index = index_entries
        self._handles = {}

        w = stats["waves"]
        p = stats["parameters"]

        # Z-score tensors for the 4 selected waveforms
        self.wave_mean = torch.tensor(
            [w[k]["mean"] for k in WAVE_KEYS_REDUCED], dtype=torch.float32
        ).unsqueeze(1)
        self.wave_std = torch.tensor(
            [w[k]["std"] for k in WAVE_KEYS_REDUCED], dtype=torch.float32
        ).unsqueeze(1)

        # Z-score scalars for Pas and Vlv (used to compute scalars)
        self._pas_mean = w["Pas"]["mean"]
        self._pas_std  = w["Pas"]["std"] + 1e-8
        self._vlv_mean = w["Vlv"]["mean"]
        self._vlv_std  = w["Vlv"]["std"] + 1e-8

        # Z-score for HR (from parameter stats)
        self._hr_mean = p["HR"]["mean"]
        self._hr_std  = p["HR"]["std"] + 1e-8

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        entry = self.index[idx]
        path = os.path.join(self.data_dir, entry["file"])
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
        g = self._handles[path][entry["group"]]

        # theta: drop HR
        theta = torch.tensor(
            [float(g[f"parameters/{k}"][()]) for k in PARAM_KEYS],
            dtype=torch.float32,
        )
        hr_raw = theta[_HR_IDX].item()
        theta_infer = torch.cat([theta[:_HR_IDX], theta[_HR_IDX + 1:]])  # (24,)

        # 4 selected waveforms, z-scored
        waves = torch.from_numpy(
            np.stack([g[f"waves/{k}"][:] for k in WAVE_KEYS_REDUCED]).astype(np.float32)
        )
        waves = (waves - self.wave_mean) / (self.wave_std + 1e-8)  # (4, 201)

        # Scalars from Pas (z-scored waveform)
        pas_z = torch.from_numpy(g["waves/Pas"][:].astype(np.float32))
        pas_z = (pas_z - self._pas_mean) / self._pas_std
        pas_mean = pas_z.mean()
        pas_max  = pas_z.max()
        pas_min  = pas_z.min()

        # SV from z-scored Vlv
        vlv_z = torch.from_numpy(g["waves/Vlv"][:].astype(np.float32))
        vlv_z = (vlv_z - self._vlv_mean) / self._vlv_std
        sv = vlv_z.max() - vlv_z.min()

        # HR z-scored
        hr_z = torch.tensor((hr_raw - self._hr_mean) / self._hr_std, dtype=torch.float32)

        scalars = torch.stack([pas_mean, pas_max, pas_min, sv, hr_z])  # (5,)

        x = torch.cat([waves.reshape(-1), scalars])  # (809,)

        return theta_infer, x

    def close(self):
        for fh in self._handles.values():
            fh.close()
        self._handles.clear()


class PairedCVDataset(Dataset):
    """Returns (theta, x_full, x_reduced) from the same simulation.

    Used in autoencoder phase 2: train ReducedAutoencoderEncoder to reconstruct
    full waveforms through the frozen phase-1 decoder.
    """

    def __init__(self, data_dir, index_entries, stats):
        self.data_dir = data_dir
        self.index    = index_entries
        self._handles = {}

        w = stats["waves"]
        p = stats["parameters"]

        self.wave_mean_full = torch.tensor(
            [w[k]["mean"] for k in WAVE_KEYS_CONT], dtype=torch.float32
        ).unsqueeze(1)
        self.wave_std_full = torch.tensor(
            [w[k]["std"] for k in WAVE_KEYS_CONT], dtype=torch.float32
        ).unsqueeze(1)

        self.wave_mean_red = torch.tensor(
            [w[k]["mean"] for k in WAVE_KEYS_REDUCED], dtype=torch.float32
        ).unsqueeze(1)
        self.wave_std_red = torch.tensor(
            [w[k]["std"] for k in WAVE_KEYS_REDUCED], dtype=torch.float32
        ).unsqueeze(1)

        self._pas_mean = w["Pas"]["mean"]
        self._pas_std  = w["Pas"]["std"] + 1e-8
        self._vlv_mean = w["Vlv"]["mean"]
        self._vlv_std  = w["Vlv"]["std"] + 1e-8
        self._hr_mean  = p["HR"]["mean"]
        self._hr_std   = p["HR"]["std"] + 1e-8

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        entry = self.index[idx]
        path  = os.path.join(self.data_dir, entry["file"])
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
        g = self._handles[path][entry["group"]]

        theta = torch.tensor(
            [float(g[f"parameters/{k}"][()]) for k in PARAM_KEYS],
            dtype=torch.float32,
        )
        hr_raw = theta[_HR_IDX].item()

        # x_full: all 28 channels flat
        waves_cont = torch.from_numpy(
            np.stack([g[f"waves/{k}"][:] for k in WAVE_KEYS_CONT]).astype(np.float32)
        )
        waves_cont = (waves_cont - self.wave_mean_full) / (self.wave_std_full + 1e-8)
        waves_valve = torch.from_numpy(
            np.stack([g[f"waves/{k}"][:] for k in WAVE_KEYS_VALVE]).astype(np.float32)
        )
        x_full = torch.cat([waves_cont, waves_valve], dim=0).reshape(-1)  # (5628,)

        # x_reduced: 4 pressure waves + 5 scalars
        waves_red = torch.from_numpy(
            np.stack([g[f"waves/{k}"][:] for k in WAVE_KEYS_REDUCED]).astype(np.float32)
        )
        waves_red = (waves_red - self.wave_mean_red) / (self.wave_std_red + 1e-8)

        pas_z = torch.from_numpy(g["waves/Pas"][:].astype(np.float32))
        pas_z = (pas_z - self._pas_mean) / self._pas_std
        vlv_z = torch.from_numpy(g["waves/Vlv"][:].astype(np.float32))
        vlv_z = (vlv_z - self._vlv_mean) / self._vlv_std
        hr_z  = torch.tensor((hr_raw - self._hr_mean) / self._hr_std, dtype=torch.float32)

        scalars   = torch.stack([pas_z.mean(), pas_z.max(), pas_z.min(),
                                 vlv_z.max() - vlv_z.min(), hr_z])
        x_reduced = torch.cat([waves_red.reshape(-1), scalars])            # (809,)

        return theta, x_full, x_reduced

    def close(self):
        for fh in self._handles.values():
            fh.close()
        self._handles.clear()


def load_stats(stats_path="norm_stats.json"):
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"{stats_path} not found. Run compute_stats.py first.")
    with open(stats_path) as f:
        return json.load(f)


def load_manifest(manifest_path):
    with open(manifest_path) as f:
        return json.load(f)
