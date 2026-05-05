# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Simulation-based inference (SBI) over cardiovascular physiology. Given observed waveforms, infer a posterior distribution over the 25 physiological parameters using the [`sbi`](https://sbi-dev.github.io/sbi/) package with the CVSurrogate MLP as the simulator.

Forked from `cv-inverse-autoencoder` (the surrogate training repo).

## Approach

- **Simulator**: `CVSurrogate` from `model.py` — takes 25 parameters → 28 waveforms (24 continuous + 4 valve), each 201 time steps. A trained checkpoint acts as a fast differentiable simulator in place of the full ODE solver.
- **Inference**: Sequential Neural Posterior Estimation (SNPE/SNLE/SNRE) via the `sbi` package.
- **PINN loss**: physics-informed regularisation term to constrain parameter estimates toward cardiovascular-consistent solutions.

## Data layout

- HDF5 files are at `DATA_DIR = /media/8TBNVME/data/neh10/hdf5/cv8/simset_10M_cv8Eed_20260314/train/`
- `manifest_train.json` lives one level above `DATA_DIR`; its `"index"` list has entries `{"id", "file", "group"}` pointing into numbered HDF5 files
- Each HDF5 group (`sim_NNNNNN`) holds `parameters/<key>` scalars and `waves/<key>` arrays of length 201
- `norm_stats.json`: normalisation stats computed by `compute_stats.py` (not committed)

## Surrogate architecture (`model.py`)

**`CVSurrogate`**:
- Trunk: 6 × Linear(1024) + SiLU
- Continuous head: Linear → reshape `(B, 24, 201)` — MSE loss
- Valve head: Linear → reshape `(B, 4, 201)` raw logits — BCEWithLogitsLoss

## Key constants

| Symbol | Value | Meaning |
|---|---|---|
| `N_PARAMS` | 25 | Input dimension |
| `N_WAVES_CONT` | 24 | Continuous output channels |
| `N_WAVES_VALVE` | 4 | Binary valve channels (av, mv, pv, tv) |
| `T` | 201 | Time steps per waveform |
| `HIDDEN` | 1024 | MLP hidden size |
| `N_LAYERS` | 6 | MLP depth |

## Workflow (planned)

1. `compute_stats.py` — compute normalisation stats (once per dataset)
2. Load a pretrained `CVSurrogate` checkpoint from `cv-inverse-autoencoder`
3. Define prior over 25 parameters
4. Run SNPE rounds using the surrogate as simulator
5. Evaluate posterior quality

## Git conventions

- Never add `Co-Authored-By: Claude` or any AI authorship trailer to commit messages.
