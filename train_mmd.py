"""
Phase 4 (MMD): Fine-tune enc_reduced via MMD domain adaptation.

Aligns real and sim latent distributions using multi-scale RBF MMD.
Real side: per-patient averaged latents (encode all beats per patient, average).
Sim side:  random batch of sim observations re-encoded each step.
MAF stays frozen throughout. Only enc_reduced is updated.

Evaluation: in a notebook, verify real waveforms reconstruct through the frozen decoder —
            that is the true test that real latents are sim-compatible.

Usage:
    python train_mmd.py
        --run exp_cnn4e64-ae-reduced_maf5_freeze-maf_1M
        --real-data ~/real_data/multibeat
        --sim-data-root /media/local/SimData/hdf5/cv8/simset_10M_cv8Eed_20260314
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy('file_system')

from dataset import ReducedCVDataset, load_stats, load_manifest
from models import ReducedAutoencoderEncoder, LATENT_DIM


STATS_PATH     = Path("norm_stats.json")
WAVE_KEYS_REAL = ["Prv", "Pra", "Pvp", "Pap"]   # matches dataset.WAVE_KEYS_REDUCED
SIM_BATCH      = 256
LR             = 1e-4
MAX_EPOCHS     = 500
PATIENCE       = 50
LOG_EVERY      = 50
N_SIM_DEFAULT  = 10_000
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


class Tee:
    def __init__(self, fh):
        self._fh = fh
        self._stdout = sys.stdout

    def write(self, msg):
        self._fh.write(msg)
        self._stdout.write(msg)

    def flush(self):
        self._fh.flush()
        self._stdout.flush()


def compute_fixed_bandwidth(z_sim: torch.Tensor, z_real: torch.Tensor = None) -> torch.Tensor:
    """Compute median pairwise distance from combined sim+real latents once at init.
    Using combined sim+real ensures the base scale captures the actual cross-distribution
    gap rather than just within-sim spacing (which was too small when real patients
    sit at the boundary of the sim cloud).
    """
    with torch.no_grad():
        z_all = torch.cat([z_sim, z_real], dim=0) if z_real is not None else z_sim
        return torch.cdist(z_all, z_all).median().clamp(min=1e-6) ** 2


def mmd_multiscale(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Multi-scale RBF MMD with per-step median heuristic bandwidth.
    Bandwidth computed from combined X+Y each call so it adapts to the actual
    cross-distribution gap. Stable because anchor loss prevents encoder drift.
    8 scales [0.1-16x] for local-to-global coverage.
    """
    with torch.no_grad():
        median_sq = torch.cdist(
            torch.cat([X, Y]), torch.cat([X, Y])
        ).median().clamp(min=1e-6) ** 2
    scales = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    loss = X.new_zeros(1)
    for scale in scales:
        gamma = 1.0 / (2.0 * scale * median_sq)
        XX = torch.exp(-gamma * torch.cdist(X, X) ** 2).mean()
        YY = torch.exp(-gamma * torch.cdist(Y, Y) ** 2).mean()
        XY = torch.exp(-gamma * torch.cdist(X, Y) ** 2).mean()
        loss = loss + (XX + YY - 2 * XY)
    return loss / len(scales)


def load_real_patients(data_dir: Path, stats: dict, log) -> list[torch.Tensor]:
    """
    Returns a list of tensors, one per patient file.
    Each tensor is (n_beats, 809) — all beats for that patient, ready to encode.
    Multibeat: one file per patient, multiple beat groups per file.
    Onebeat:   one beat per file — each file treated as its own patient.
    """
    w = stats["waves"]
    p = stats["parameters"]

    wave_mean = torch.tensor(
        [w[k]["mean"] for k in WAVE_KEYS_REAL], dtype=torch.float32
    ).unsqueeze(1)
    wave_std = torch.tensor(
        [w[k]["std"] for k in WAVE_KEYS_REAL], dtype=torch.float32
    ).unsqueeze(1)
    pas_mean = w["Pas"]["mean"];  pas_std = w["Pas"]["std"] + 1e-8
    vlv_std  = w["Vlv"]["std"] + 1e-8
    hr_mean  = p["HR"]["mean"];   hr_std  = p["HR"]["std"]  + 1e-8

    patient_tensors = []
    for fpath in sorted(data_dir.glob("*.h5")):
        beats = []
        with h5py.File(fpath, "r") as f:
            for beat_key in sorted(f.keys()):
                if not beat_key.startswith("beat_"):
                    continue
                g = f[beat_key]
                waves = np.stack(
                    [g[f"waves/{k}"][:].astype(np.float32) for k in WAVE_KEYS_REAL]
                )
                waves_t = (torch.from_numpy(waves) - wave_mean) / (wave_std + 1e-8)

                sbp  = float(g["summaries/sbp"][()])
                dbp  = float(g["summaries/dbp"][()])
                map_ = float(g["summaries/map"][()])
                sv   = float(g["summaries/sv"][()])
                hr   = float(g["parameters/HR"][()])

                scalars = torch.tensor([
                    (map_ - pas_mean) / pas_std,
                    (sbp  - pas_mean) / pas_std,
                    (dbp  - pas_mean) / pas_std,
                    sv   / vlv_std,
                    (hr   - hr_mean)  / hr_std,
                ], dtype=torch.float32)

                beats.append(torch.cat([waves_t.reshape(-1), scalars]))  # (809,)

        if beats:
            patient_tensors.append(torch.stack(beats))  # (n_beats, 809)

    n_patients = len(patient_tensors)
    n_beats    = sum(t.shape[0] for t in patient_tensors)
    log(f"Loaded {n_patients} patients, {n_beats} beats total from {data_dir.name}")
    return patient_tensors


def patient_averaged_latents(enc, patient_tensors: list[torch.Tensor]) -> torch.Tensor:
    """Encode all beats per patient, average latents. Returns (n_patients, latent_dim)."""
    enc.eval()
    averaged = []
    with torch.no_grad():
        for beats in patient_tensors:
            z = enc(beats.to(DEVICE))      # (n_beats, latent_dim)
            averaged.append(z.mean(dim=0)) # (latent_dim,)
    return torch.stack(averaged)           # (n_patients, latent_dim)


def load_sim_data(data_dir: Path, manifest: dict, stats: dict, n: int, log) -> torch.Tensor:
    index   = manifest["index"][:n]
    dataset = ReducedCVDataset(str(data_dir), index, stats)
    loader  = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)
    xs, loaded = [], 0
    for _, x_batch in loader:
        xs.append(x_batch)
        loaded += len(x_batch)
        print(f"\r  loaded {loaded}/{n}", end="", flush=True)
    print()
    dataset.close()
    x_sim = torch.cat(xs)
    log(f"Loaded {loaded} sim observations  x={tuple(x_sim.shape[1:])}")
    return x_sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",           required=True,
                        help="Base ae-reduced run to fine-tune")
    parser.add_argument("--real-data",     required=True,
                        help="Real data directory (multibeat recommended)")
    parser.add_argument("--output-run",    required=True,
                        help="Output run name, e.g. exp_..._mmd-multibeat-fixed-bw")
    parser.add_argument("--sim-data-root", required=True,
                        help="Dataset root with train/ and manifest_train.json")
    parser.add_argument("--n-sim",         type=int,   default=N_SIM_DEFAULT)
    parser.add_argument("--lr",            type=float, default=LR)
    parser.add_argument("--weight-decay",  type=float, default=1e-4)
    parser.add_argument("--grad-clip",     type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=1.0,
                        help="Weight for sim anchor loss: λ * ||enc(x_sim) - enc_frozen(x_sim)||²")
    parser.add_argument("--epochs",   type=int,   default=MAX_EPOCHS,
                        help="Training epochs (= gradient steps since all real patients used each epoch)")
    parser.add_argument("--patience", type=int,   default=PATIENCE)
    parser.add_argument("--dry-run",  action="store_true",
                        help="Smoke test: 2 epochs, 512 sim samples, no posterior saved")
    args = parser.parse_args()

    if args.dry_run:
        args.epochs   = 2
        args.n_sim    = 512
        args.patience = 999

    real_data_name = Path(args.real_data).name
    mmd_run_name   = args.output_run
    base_run_dir   = Path("outputs") / args.run
    out_root       = Path("dry-runs") if args.dry_run else Path("outputs")
    mmd_run_dir    = out_root / mmd_run_name
    mmd_run_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = mmd_run_dir / f"mmd_{mmd_run_name}_{date_str}.log"
    log_fh   = open(log_path, "w")
    _stdout, _stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(log_fh)
    sys.stderr = Tee(log_fh)

    def log(msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"MMD fine-tuning: {args.run} → {mmd_run_name}  ({'DRY RUN' if args.dry_run else 'full'})")
    log(f"Device: {DEVICE}  lr: {args.lr}  max_epochs: {args.epochs}  patience: {args.patience}")
    log(f"Real data: {args.real_data}  n_sim: {args.n_sim}")

    stats    = load_stats(STATS_PATH)
    manifest = load_manifest(Path(args.sim_data_root) / "manifest_train.json")

    patient_tensors = load_real_patients(Path(args.real_data), stats, log)

    log(f"Loading {args.n_sim} sim observations into RAM...")
    x_sim = load_sim_data(
        Path(args.sim_data_root) / "train", manifest, stats, args.n_sim, log
    )

    ckpt = base_run_dir / "phase2_encoder.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found")
    enc = ReducedAutoencoderEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    enc.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    log(f"Loaded enc_reduced from {ckpt}")

    # Frozen reference encoder — pins sim latents to their initial positions
    enc_frozen = ReducedAutoencoderEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    enc_frozen.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    enc_frozen.requires_grad_(False)
    enc_frozen.eval()

    # Log initial MMD
    with torch.no_grad():
        z_sim_init  = enc(x_sim[:1000].to(DEVICE))
        z_real_init = patient_averaged_latents(enc, patient_tensors)
    log(f"Initial MMD: {mmd_multiscale(z_real_init, z_sim_init).item():.4f}")

    opt = torch.optim.Adam(enc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    log(f"Optimiser: Adam lr={args.lr}  weight_decay={args.weight_decay}  grad_clip={args.grad_clip}")

    best_mmd  = float("inf")
    wait      = 0
    best_state = {k: v.clone() for k, v in enc.state_dict().items()}

    log("Training...")
    for epoch in range(1, args.epochs + 1):
        enc.train()

        # Real side: all 60 patients, averaged latents (re-encoded each epoch)
        z_real = patient_averaged_latents(enc, patient_tensors)

        # Sim side: random batch of 256, re-encoded each epoch
        idx   = torch.randperm(len(x_sim))[:SIM_BATCH]
        z_sim = enc(x_sim[idx].to(DEVICE))

        mmd_loss = mmd_multiscale(z_real, z_sim)
        with torch.no_grad():
            z_sim_frozen = enc_frozen(x_sim[idx].to(DEVICE))
        anchor_loss = torch.nn.functional.mse_loss(z_sim, z_sim_frozen)
        loss = mmd_loss + args.anchor_weight * anchor_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), args.grad_clip)
        opt.step()

        mmd_val = mmd_loss.item()

        if mmd_val < best_mmd:
            best_mmd   = mmd_val
            wait       = 0
            best_state = {k: v.clone() for k, v in enc.state_dict().items()}
        else:
            wait += 1

        if epoch % LOG_EVERY == 0 or epoch == 1:
            log(f"  epoch {epoch:4d}/{args.epochs}  mmd={mmd_val:.4f}  anchor={anchor_loss.item():.4f}  best={best_mmd:.4f}  wait={wait}")

        if wait >= args.patience:
            log(f"  early stop at epoch {epoch}  best_mmd={best_mmd:.4f}")
            break

    # Restore best checkpoint
    enc.load_state_dict(best_state)
    torch.save(enc.state_dict(), mmd_run_dir / "mmd_encoder.pt")
    log("Saved mmd_encoder.pt (best checkpoint)")

    if args.dry_run:
        log("Dry run — posterior not saved.")
    else:
        log("Updating posterior with fine-tuned encoder...")
        posterior = torch.load(
            base_run_dir / "posterior.pt", map_location=DEVICE, weights_only=False
        )
        posterior.posterior_estimator.embedding_net.load_state_dict(enc.state_dict())
        torch.save(posterior, mmd_run_dir / "posterior.pt")
        log(f"Saved posterior to {mmd_run_dir / 'posterior.pt'}")

    mmd_info = dict(
        run=mmd_run_name,
        base_run=args.run,
        real_data=str(args.real_data),
        real_data_name=real_data_name,
        n_patients=len(patient_tensors),
        n_beats=sum(t.shape[0] for t in patient_tensors),
        timestamp=datetime.now().isoformat(timespec="seconds"),
        git_hash=git_hash(),
        device=DEVICE,
        training=dict(
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            anchor_weight=args.anchor_weight,
            max_epochs=args.epochs,
            patience=args.patience,
            sim_batch=SIM_BATCH,
            n_sim=args.n_sim,
            best_mmd=best_mmd,
            kernel_scales=[0.25, 0.5, 1.0, 2.0, 4.0],
        ),
    )
    (mmd_run_dir / "mmd_info.json").write_text(json.dumps(mmd_info, indent=2))
    log("Saved mmd_info.json")

    log_fh.close()
    sys.stdout = _stdout
    sys.stderr = _stderr
    print(f"\nLog written to {log_path}")


if __name__ == "__main__":
    main()
