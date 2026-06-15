# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Simulation-based inference (SBI) over cardiovascular physiology. Given observed waveforms, infer a posterior distribution over the 25 physiological parameters using the [`sbi`](https://sbi-dev.github.io/sbi/) package with pre-simulated (theta, waveform) pairs.

Forked from `cv-inverse-autoencoder` (the surrogate training repo).

## Approach

- **No live simulator**: uses pre-simulated HDF5 data directly via `sbi`'s `append_simulations`
- **Inference**: Neural Posterior Estimation (NPE) via the `sbi` package (v0.26)
- **Observation**: 28 waveforms × 201 time steps per simulation

## Data layout

- HDF5 files under `<data-root>/train/` and `<data-root>/test/`; pass the root via `--data-root`
  - Adamant: `/media/pulsar/SimData/hdf5/cv8/simset_10M_cv8Eed_20260314`
- `manifest_train.json` / `manifest_test.json` live at `<data-root>/` (one level above the data dirs)
- Each HDF5 group (`sim_NNNNNN`) holds `parameters/<key>` scalars and `waves/<key>` arrays of length 201
- `norm_stats.json`: wave normalisation stats (not committed)
- 25 variable parameters defined by `pvar_low`/`pvar_high` in manifest config

## Key constants

| Symbol | Value | Meaning |
|---|---|---|
| `N_PARAMS` | 25 | Parameters to infer |
| `N_CHANNELS` | 28 | Total waveform channels (24 continuous + 4 valve) |
| `T` | 201 | Time steps per waveform |

## Run naming convention

NPE runs (train_sbi.py): `{type}_{series}_{embedding}_{flow}_{training-method}`
Domain adaptation runs (train_mmd.py, train_ot.py): `{type}_{series}_{embedding}_{1M}_{method}`

- type: `exp` (full run) or `dry` (512 sims, smoke test, no posterior saved)
- embedding: `cnn4e64` (4-block CNN, 64-dim), `sumstats` (hand-crafted summary stats), etc.
- flow (NPE only): `maf5` (MAF, 5 transforms), `nsf8`, etc.
- training-method (NPE only): describes what is frozen/ablated, e.g. `freeze-maf`, `freeze-input`
- method (domain adaptation): describes the adaptation approach, e.g. `mmd-multibeat-adaptive`, `ot-sinkhorn-warm`

Examples:
- NPE: `exp_cnn4e64-ae-reduced_maf5_freeze-maf_1M`
- MMD: `exp_cnn4e64-ae-reduced_1M_mmd-multibeat-adaptive`
- OT:  `exp_cnn4e64-ae-reduced_1M_ot-sinkhorn-warm`

## Experiment tracking

- `outputs/{run_name}/` — full runs (gitignored)
- `dry-runs/{run_name}/` — dry runs (gitignored)
- Each run writes `run_info.json` (git hash, config, architecture) and `train_log.txt`
- **Always commit before starting an `exp_` run** so `run_info.json` captures the exact code
- **Always confirm the output run name/directory with the user before executing any training run** — never assume the name is correct, especially for variant runs that could overwrite existing results.

## Git conventions

- Never add `Co-Authored-By: Claude` or any AI authorship trailer to commit messages.
- Always commit before running a full experiment.
