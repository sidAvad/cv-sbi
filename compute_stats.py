"""
Compute per-channel normalisation statistics (mean, std) from a single HDF5
batch file and save to norm_stats.json.

Loads all 10k sims from 1.h5 into numpy arrays, then just calls .mean() and
.std(). Simple and fast enough — no online algorithms needed at this scale.

Run this once before train.py.
"""

import json
from pathlib import Path

import h5py
import numpy as np


# ─── Config ──────────────────────────────────────────────────────────────────

DATA_DIR = Path(
    "/media/8TBNVME/data/neh10/hdf5/cv8/simset_10M_cv8Eed_20260314/train"
)
SOURCE_FILE = "1.h5"
STATS_PATH = Path("norm_stats.json")

PARAM_KEYS = [
    "AVD", "Bla", "Blv", "Bra", "Brv",
    "Cas", "Cvp", "Cvs", "Eap",
    "Eedref_la", "Eedref_lv", "Eedref_ra", "Eedref_rv",
    "Emax_LA", "Emax_LV", "Emax_RA", "Emax_RV",
    "HR", "Rap", "Ras", "Tmax", "Tmax_a",
    "Vs", "τ", "τ_a",
]  # 25 varying parameters

WAVE_KEYS_CONT = [
    "Pap", "Pas", "Pla", "Plv", "Pra", "Prv", "Pvp", "Pvs",
    "Qap", "Qas", "Qla", "Qlv", "Qra", "Qrv", "Qvp", "Qvs",
    "Vap", "Vas", "Vla", "Vlv", "Vra", "Vrv", "Vvp", "Vvs",
]  # 24 continuous waveforms


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    src = DATA_DIR / SOURCE_FILE
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    print(f"Reading {src}...")
    with h5py.File(src, "r") as f:
        sim_groups = sorted(
            k for k in f.keys()
            if isinstance(f[k], h5py.Group) and k.startswith("sim_")
        )
        n_sims = len(sim_groups)
        print(f"  Found {n_sims} simulations")

        # Stack all sims: params (n_sims, n_params), waves (n_sims, n_waves, T)
        params = np.stack([
            [float(f[sim][f"parameters/{k}"][()]) for k in PARAM_KEYS]
            for sim in sim_groups
        ]).astype(np.float64)

        waves = np.stack([
            np.stack([f[sim][f"waves/{k}"][:] for k in WAVE_KEYS_CONT])
            for sim in sim_groups
        ]).astype(np.float64)

    print(f"  params shape: {params.shape}")
    print(f"  waves  shape: {waves.shape}")

    # ── Compute stats ──────────────────────────────────────────────────
    # params: one value per sim per channel → reduce over sim axis
    p_mean = params.mean(axis=0)
    p_std = params.std(axis=0)

    # waves: reduce over sim and time axes (keep channel axis)
    w_mean = waves.mean(axis=(0, 2))
    w_std = waves.std(axis=(0, 2))

    # ── Assemble and save ──────────────────────────────────────────────
    stats = {
        "parameters": {
            k: {"mean": float(p_mean[i]), "std": float(p_std[i])}
            for i, k in enumerate(PARAM_KEYS)
        },
        "waves": {
            k: {"mean": float(w_mean[i]), "std": float(w_std[i])}
            for i, k in enumerate(WAVE_KEYS_CONT)
        },
        "source_file": SOURCE_FILE,
        "n_sims_used": n_sims,
    }

    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # ── Print summary ──────────────────────────────────────────────────
    print(f"\nSaved stats to {STATS_PATH.resolve()}")

    print("\nParameter stats:")
    print(f"  {'name':<15} {'mean':>14} {'std':>14}")
    for k, v in stats["parameters"].items():
        print(f"  {k:<15} {v['mean']:>14.4f} {v['std']:>14.4f}")

    print("\nWaveform stats:")
    print(f"  {'name':<15} {'mean':>14} {'std':>14}")
    for k, v in stats["waves"].items():
        print(f"  {k:<15} {v['mean']:>14.4f} {v['std']:>14.4f}")


if __name__ == "__main__":
    main()