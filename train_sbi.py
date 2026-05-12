"""
Train a Neural Posterior Estimator (NPE) over 25 cardiovascular parameters
using pre-simulated (theta, waveform) pairs from HDF5 files.

Usage:
    python train_sbi.py --run exp_baseline
    python train_sbi.py --run dry-run_smoke
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform

from dataset import CVDataset, PARAM_KEYS, N_CHANNELS, T, load_stats, load_manifest


# ─── Config ──────────────────────────────────────────────────────────────────

DATA_DIR    = Path("/media/8TBNVME/data/neh10/hdf5/cv8/simset_10M_cv8Eed_20260314/train")
MANIFEST    = DATA_DIR.parent / "manifest_train.json"
STATS_PATH  = Path("norm_stats.json")

N_SIMS_FULL    = 100_000
N_SIMS_DRYRUN  = 512

BATCH_SIZE      = 512
EMBED_DIM       = 64
HIDDEN_FEATURES = 128
NUM_TRANSFORMS  = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Embedding net ────────────────────────────────────────────────────────────

class WaveformEmbedding(nn.Module):
    """1D CNN: flat (28*201,) → embed_dim."""

    # Each entry: (in_ch, out_ch, kernel, stride)
    CONV_LAYERS = [
        (N_CHANNELS, 64,  7, 1),
        (64,         128, 5, 2),
        (128,        256, 5, 2),
        (256,        256, 3, 1),
    ]

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.n_channels = N_CHANNELS
        self.t = T
        self.embed_dim = embed_dim

        layers = []
        for in_ch, out_ch, k, s in self.CONV_LAYERS:
            layers += [nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2, stride=s), nn.SiLU()]
        self.cnn = nn.Sequential(*layers)
        self.proj = nn.Linear(self.CONV_LAYERS[-1][1], embed_dim)

    def describe(self) -> dict:
        return {
            "type": "WaveformEmbedding",
            "input": f"({self.n_channels}, {self.t})",
            "conv_layers": [
                {"in": ic, "out": oc, "kernel": k, "stride": s}
                for ic, oc, k, s in self.CONV_LAYERS
            ],
            "pooling": "global_avg",
            "embed_dim": self.embed_dim,
            "n_params": sum(p.numel() for p in self.parameters()),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.n_channels, self.t)
        h = self.cnn(x).mean(dim=-1)
        return self.proj(h)


# ─── Prior ────────────────────────────────────────────────────────────────────

def build_prior(manifest: dict) -> BoxUniform:
    lo = manifest["config"]["pvar_low"]
    hi = manifest["config"]["pvar_high"]
    return BoxUniform(
        low=torch.tensor([lo[k] for k in PARAM_KEYS], dtype=torch.float32),
        high=torch.tensor([hi[k] for k in PARAM_KEYS], dtype=torch.float32),
        device=DEVICE,
    )


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_simulations(manifest: dict, stats: dict, n: int, log):
    index = manifest["index"][:n]
    dataset = CVDataset(str(DATA_DIR), index, stats)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    thetas, xs = [], []
    loaded = 0
    for theta_batch, x_batch in loader:
        thetas.append(theta_batch)
        xs.append(x_batch)
        loaded += len(theta_batch)
        print(f"\r  loaded {loaded}/{n}", end="", flush=True)
    print()
    log(f"Loaded {loaded} simulations  theta={tuple(thetas[0].shape[1:])}  x={tuple(xs[0].shape[1:])}")

    dataset.close()
    return torch.cat(thetas), torch.cat(xs)


# ─── Run management ───────────────────────────────────────────────────────────

def parse_run(name: str):
    if name.startswith("dry-run_"):
        return "dry-run", name, Path("dry-runs") / name
    elif name.startswith("exp_"):
        return "exp", name, Path("outputs") / name
    else:
        raise ValueError("--run must start with 'exp_' or 'dry-run_'")


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


class Tee:
    """Write to both a file and the original stdout."""
    def __init__(self, fh):
        self._fh = fh
        self._stdout = sys.stdout

    def write(self, msg):
        self._fh.write(msg)
        self._stdout.write(msg)

    def flush(self):
        self._fh.flush()
        self._stdout.flush()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="e.g. exp_baseline or dry-run_smoke")
    args = parser.parse_args()

    run_type, run_name, run_dir = parse_run(args.run)
    is_dry = run_type == "dry-run"
    n_sims = N_SIMS_DRYRUN if is_dry else N_SIMS_FULL

    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "train_log.txt"
    log_fh = open(log_path, "w")
    _real_stdout = sys.stdout
    sys.stdout = Tee(log_fh)

    def log(msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"Run: {run_name}  ({'dry-run' if is_dry else 'full'})")
    log(f"Device: {DEVICE}")

    embedding = WaveformEmbedding()

    run_info = dict(
        run=run_name,
        type=run_type,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        script=str(Path(sys.argv[0]).resolve()),
        git_hash=git_hash(),
        device=DEVICE,
        data=dict(
            n_sims=n_sims,
            data_dir=str(DATA_DIR),
            manifest=str(MANIFEST),
        ),
        embedding=embedding.describe(),
        flow=dict(
            model="maf",
            hidden_features=HIDDEN_FEATURES,
            num_transforms=NUM_TRANSFORMS,
            z_score_theta="independent",
            z_score_x="none",
        ),
        training=dict(
            batch_size=BATCH_SIZE,
        ),
    )
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    log(f"run_info.json written  git={run_info['git_hash']}")

    log("Loading manifest and stats...")
    manifest = load_manifest(MANIFEST)
    stats    = load_stats(STATS_PATH)

    log(f"Loading {n_sims:,} simulations...")
    theta, x = load_simulations(manifest, stats, n_sims, log)

    prior = build_prior(manifest)

    density_estimator_fn = posterior_nn(
        model="maf",
        embedding_net=embedding,
        hidden_features=HIDDEN_FEATURES,
        num_transforms=NUM_TRANSFORMS,
        z_score_theta="independent",
        z_score_x="none",
    )

    inference = NPE(prior=prior, density_estimator=density_estimator_fn, device=DEVICE)
    inference.append_simulations(theta, x)

    log("Training NPE...")
    density_estimator = inference.train(
        training_batch_size=BATCH_SIZE,
        show_train_summary=True,
    )

    if not is_dry:
        log("Building and saving posterior...")
        posterior = inference.build_posterior(density_estimator)
        torch.save(posterior, run_dir / "posterior.pt")
        log(f"Saved posterior to {run_dir / 'posterior.pt'}")
    else:
        log("Dry-run complete — posterior not saved.")

    log_fh.close()
    sys.stdout = _real_stdout
    print(f"\nLog written to {log_path}")


if __name__ == "__main__":
    main()
