"""
PyTorch Dataset for SBI over cardiovascular physiology.

Returns raw (unnormalized) parameters as theta, and all 28 waveforms
(24 continuous z-scored + 4 binary valves) stacked as a flat observation
vector for sbi's append_simulations.
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


def load_stats(stats_path="norm_stats.json"):
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"{stats_path} not found. Run compute_stats.py first.")
    with open(stats_path) as f:
        return json.load(f)


def load_manifest(manifest_path):
    with open(manifest_path) as f:
        return json.load(f)
