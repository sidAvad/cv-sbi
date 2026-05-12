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
