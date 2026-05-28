"""
Train a Neural Posterior Estimator (NPE) over 25 cardiovascular parameters
using pre-simulated (theta, waveform) pairs from HDF5 files.

Usage:
    python train_sbi.py --run exp_baseline_cnn4e64_maf5 --data-root /media/pulsar/SimData/hdf5/cv8/simset_10M_cv8Eed_20260314
    python train_sbi.py --run dry_sumstats_maf5 --data-root /path/to/dataset

--data-root should point to the dataset root directory, which must contain
train/ and manifest_train.json.
"""

import argparse
import json
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

from dataset import (
    CVDataset, ReducedCVDataset,
    PARAM_KEYS, PARAM_KEYS_INFER,
    N_CHANNELS, T,
    N_REDUCED_CHANNELS, N_SCALARS, OBS_DIM,
    load_stats, load_manifest,
    compute_summary_stats, N_SUMSTATS,
)


# ─── Config ──────────────────────────────────────────────────────────────────

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
    """1D CNN: flat (28*201,) → embed_dim. pooling: 'attention' or 'mean'."""

    # Each entry: (in_ch, out_ch, kernel, stride)
    CONV_LAYERS = [
        (N_CHANNELS, 64,  7, 1),
        (64,         128, 5, 2),
        (128,        256, 5, 2),
        (256,        256, 3, 1),
    ]

    def __init__(self, embed_dim: int = EMBED_DIM, pooling: str = "attention"):
        super().__init__()
        self.n_channels = N_CHANNELS
        self.t = T
        self.embed_dim = embed_dim
        self.pooling = pooling

        layers = []
        for in_ch, out_ch, k, s in self.CONV_LAYERS:
            layers += [nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2, stride=s), nn.SiLU()]
        self.cnn = nn.Sequential(*layers)
        if pooling == "attention":
            self.attn_pool = nn.Linear(self.CONV_LAYERS[-1][1], 1)
        self.proj = nn.Linear(self.CONV_LAYERS[-1][1], embed_dim)

    def describe(self) -> dict:
        return {
            "type": "WaveformEmbedding",
            "input": f"({self.n_channels}, {self.t})",
            "conv_layers": [
                {"in": ic, "out": oc, "kernel": k, "stride": s}
                for ic, oc, k, s in self.CONV_LAYERS
            ],
            "pooling": self.pooling,
            "embed_dim": self.embed_dim,
            "n_params": sum(p.numel() for p in self.parameters()),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.n_channels, self.t)
        h = self.cnn(x).transpose(1, 2)              # (B, T', 256)
        if self.pooling == "attention":
            w = self.attn_pool(h).softmax(dim=1)      # (B, T', 1)
            h = (w * h).sum(dim=1)                    # (B, 256)
        else:
            h = h.mean(dim=1)                         # (B, 256)
        return self.proj(h)


# ─── Reduced embedding net ───────────────────────────────────────────────────

class ReducedWaveformEmbedding(nn.Module):
    """4-channel pressure CNN + 5 scalars as prefix tokens → attention pool → embed_dim."""

    CONV_LAYERS = [
        (N_REDUCED_CHANNELS, 64,  7, 1),
        (64,                 128, 5, 2),
        (128,                256, 5, 2),
        (256,                256, 3, 1),
    ]

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        self.wave_len = N_REDUCED_CHANNELS * T  # 804
        feat_dim = self.CONV_LAYERS[-1][1]       # 256

        layers = []
        for in_ch, out_ch, k, s in self.CONV_LAYERS:
            layers += [nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2, stride=s), nn.SiLU()]
        self.cnn = nn.Sequential(*layers)
        self.scalar_projs = nn.ModuleList([nn.Linear(1, feat_dim) for _ in range(N_SCALARS)])
        self.attn_pool = nn.Linear(feat_dim, 1)
        self.proj = nn.Linear(feat_dim, embed_dim)

    @property
    def output_dim(self):
        return self.embed_dim

    def describe(self) -> dict:
        return {
            "type": "ReducedWaveformEmbedding",
            "input_waveforms": f"({N_REDUCED_CHANNELS}, {T})",
            "input_scalars": "Pas_mean, Pas_max, Pas_min, SV, HR_z",
            "scalar_integration": "prefix_tokens_before_attn_pool",
            "conv_layers": [
                {"in": ic, "out": oc, "kernel": k, "stride": s}
                for ic, oc, k, s in self.CONV_LAYERS
            ],
            "pooling": "attention",
            "embed_dim": self.embed_dim,
            "output_dim": self.output_dim,
            "n_params": sum(p.numel() for p in self.parameters()),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        waves   = x[:, :self.wave_len].view(-1, N_REDUCED_CHANNELS, T)
        scalars = x[:, self.wave_len:]                                    # (B, 5)

        h = self.cnn(waves).transpose(1, 2)                               # (B, T', 256)
        scalar_tokens = torch.stack(
            [proj(scalars[:, i:i+1]) for i, proj in enumerate(self.scalar_projs)],
            dim=1,
        )                                                                  # (B, 5, 256)
        h = torch.cat([scalar_tokens, h], dim=1)                          # (B, T'+5, 256)
        w = self.attn_pool(h).softmax(dim=1)                              # (B, T'+5, 1)
        h = (w * h).sum(dim=1)                                            # (B, 256)
        return self.proj(h)


# ─── Prior ────────────────────────────────────────────────────────────────────

def build_prior(manifest: dict, param_keys: list) -> BoxUniform:
    lo = manifest["config"]["pvar_low"]
    hi = manifest["config"]["pvar_high"]
    return BoxUniform(
        low=torch.tensor([lo[k] for k in param_keys], dtype=torch.float32),
        high=torch.tensor([hi[k] for k in param_keys], dtype=torch.float32),
        device=DEVICE,
    )


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_simulations(data_dir: Path, manifest: dict, stats: dict, n: int, log, dataset_cls):
    index = manifest["index"][:n]
    dataset = dataset_cls(str(data_dir), index, stats)
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
    if name.startswith("dry_"):
        return "dry", name, Path("dry-runs") / name
    elif name.startswith("exp_"):
        return "exp", name, Path("outputs") / name
    else:
        raise ValueError("--run must start with 'exp_' or 'dry_'")


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
    parser.add_argument("--run", required=True,
                        help="e.g. exp_cnn4e64_maf5_freeze-maf or dry_cnn4e64_maf5")
    parser.add_argument("--data-root", required=True,
                        help="Dataset root containing train/ and manifest_train.json")
    parser.add_argument("--embedding", choices=["cnn", "cnn-meanpool", "sumstats", "cnn-reduced"],
                        default="cnn",
                        help="cnn: full 28-ch CNN attn pool; cnn-meanpool: full 28-ch CNN mean pool; cnn-reduced: 4 pressure waves + scalars; sumstats: hand-crafted")
    parser.add_argument("--freeze-epochs", type=int, default=0,
                        help="Epochs to train embedding only before joint training (0 = no freeze)")
    args = parser.parse_args()

    run_type, run_name, run_dir = parse_run(args.run)
    is_dry       = run_type == "dry"
    use_reduced  = args.embedding == "cnn-reduced"
    use_sumstats = args.embedding == "sumstats"
    use_meanpool = args.embedding == "cnn-meanpool"
    data_root   = Path(args.data_root)
    data_dir    = data_root / "train"
    manifest_path = data_root / "manifest_train.json"
    n_sims = N_SIMS_DRYRUN if is_dry else N_SIMS_FULL

    run_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = run_dir / f"train_{run_name}_{date_str}.log"
    log_fh   = open(log_path, "w")
    _real_stdout = sys.stdout
    _real_stderr = sys.stderr
    sys.stdout = Tee(log_fh)
    sys.stderr = Tee(log_fh)

    def log(msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"Run: {run_name}  ({'dry' if is_dry else 'full'})")
    log(f"Device: {DEVICE}  embedding: {args.embedding}")

    if use_reduced:
        embedding_net = ReducedWaveformEmbedding()
        embedding_info = embedding_net.describe()
        z_score_x  = "none"
        dataset_cls = ReducedCVDataset
        param_keys  = PARAM_KEYS_INFER
    elif use_sumstats:
        embedding_net = nn.Identity()
        embedding_info = {
            "type": "sumstats", "n_features": N_SUMSTATS,
            "features": "pressure(mean,sys,dia,pp)*8 + flow(mean,peak,min)*8 + volume(EDV,ESV,SV)*8 + valve(open_frac)*4",
        }
        z_score_x   = "independent"
        dataset_cls = CVDataset
        param_keys  = PARAM_KEYS
    elif use_meanpool:
        embedding_net = WaveformEmbedding(pooling="mean")
        embedding_info = embedding_net.describe()
        z_score_x   = "none"
        dataset_cls = CVDataset
        param_keys  = PARAM_KEYS
    else:
        embedding_net = WaveformEmbedding()
        embedding_info = embedding_net.describe()
        z_score_x   = "none"
        dataset_cls = CVDataset
        param_keys  = PARAM_KEYS

    run_info = dict(
        run=run_name,
        type=run_type,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        script=str(Path(sys.argv[0]).resolve()),
        git_hash=git_hash(),
        device=DEVICE,
        data=dict(
            n_sims=n_sims,
            data_dir=str(data_dir),
            manifest=str(manifest_path),
        ),
        embedding=embedding_info,
        flow=dict(
            model="maf",
            hidden_features=HIDDEN_FEATURES,
            num_transforms=NUM_TRANSFORMS,
            z_score_theta="independent",
            z_score_x=z_score_x,
        ),
        training=dict(
            batch_size=BATCH_SIZE,
            freeze_epochs=args.freeze_epochs,
        ),
    )
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    log(f"run_info.json written  git={run_info['git_hash']}")

    log("Loading manifest and stats...")
    manifest = load_manifest(manifest_path)
    stats    = load_stats(STATS_PATH)

    log(f"Loading {n_sims:,} simulations...")
    theta, x = load_simulations(data_dir, manifest, stats, n_sims, log, dataset_cls)

    if use_sumstats:
        log("Computing summary statistics...")
        x = compute_summary_stats(x)
        log(f"Summary stats shape: {tuple(x.shape)}")

    prior = build_prior(manifest, param_keys)

    density_estimator_fn = posterior_nn(
        model="maf",
        embedding_net=embedding_net,
        hidden_features=HIDDEN_FEATURES,
        num_transforms=NUM_TRANSFORMS,
        z_score_theta="independent",
        z_score_x=z_score_x,
    )

    inference = NPE(prior=prior, density_estimator=density_estimator_fn, device=DEVICE)
    inference.append_simulations(theta, x)

    if args.freeze_epochs > 0:
        log("Initialising network (1 epoch, all params)...")
        inference.train(max_num_epochs=1, training_batch_size=BATCH_SIZE,
                        show_train_summary=False)

        log(f"Phase 1: freeze MAF, train embedding for {args.freeze_epochs} epochs...")
        for name, p in inference._neural_net.named_parameters():
            if "embedding_net" not in name:
                p.requires_grad_(False)
        inference.train(resume_training=True, max_num_epochs=args.freeze_epochs,
                        training_batch_size=BATCH_SIZE, show_train_summary=False)

        log("Phase 2: joint training (MAF unfrozen, sbi early stopping)...")
        for p in inference._neural_net.parameters():
            p.requires_grad_(True)

    else:
        log("Training NPE...")

    density_estimator = inference.train(
        resume_training=args.freeze_epochs > 0,
        training_batch_size=BATCH_SIZE,
        show_train_summary=True,
    )

    if not is_dry:
        log("Building and saving posterior...")
        posterior = inference.build_posterior(density_estimator)
        torch.save(posterior, run_dir / "posterior.pt")
        log(f"Saved posterior to {run_dir / 'posterior.pt'}")
    else:
        log("Dry run complete — posterior not saved.")

    log_fh.close()
    sys.stdout = _real_stdout
    sys.stderr = _real_stderr
    print(f"\nLog written to {log_path}")


if __name__ == "__main__":
    main()
