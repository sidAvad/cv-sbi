"""
Phases 1 and 2 of autoencoder pre-training for cv-sbi.

Phase 1: AutoencoderEncoder (full 28-ch CNN) + WaveformDecoder trained to
         reconstruct all 28 waveforms from a 128-dim latent (MSE loss, 30 epochs).
         Saves: phase1_encoder.pt, phase1_decoder.pt

Phase 2: Load phase1_decoder.pt and freeze it. Train ReducedAutoencoderEncoder
         (4-ch pressure CNN + 5 scalar prefix tokens → 128-dim) to reconstruct
         full waveforms through the frozen decoder (MSE loss, 30 epochs).
         Saves: phase2_encoder.pt

After this script, run train_sbi.py --embedding ae-reduced to execute phases 3+4.

Usage:
    python train_autoencoder.py --run exp_cnn4e64-ae-reduced_maf5_freeze-maf --data-root /path/to/data
    python train_autoencoder.py --run dry_cnn4e64-ae-reduced_maf5_freeze-maf --data-root /path/to/data
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

from dataset import (
    CVDataset, PairedCVDataset,
    load_stats, load_manifest,
)
from models import AutoencoderEncoder, ReducedAutoencoderEncoder, WaveformDecoder, LATENT_DIM


STATS_PATH    = Path("norm_stats.json")
N_SIMS_FULL   = 100_000
N_SIMS_DRY    = 512
BATCH_SIZE    = 512
PHASE1_EPOCHS = 30
PHASE2_EPOCHS = 30
LR            = 1e-3
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


class Tee:
    def __init__(self, fh):
        self._fh     = fh
        self._stdout = sys.stdout

    def write(self, msg):
        self._fh.write(msg)
        self._stdout.write(msg)

    def flush(self):
        self._fh.flush()
        self._stdout.flush()


def parse_run(name):
    if name.startswith("dry_"):
        return "dry", name, Path("dry-runs") / name
    elif name.startswith("exp_"):
        return "exp", name, Path("outputs") / name
    else:
        raise ValueError("--run must start with 'exp_' or 'dry_'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",       required=True,
                        help="e.g. exp_cnn4e64-ae-reduced_maf5_freeze-maf")
    parser.add_argument("--data-root", required=True,
                        help="Dataset root containing train/ and manifest_train.json")
    args = parser.parse_args()

    run_type, run_name, run_dir = parse_run(args.run)
    is_dry    = run_type == "dry"
    n_sims    = N_SIMS_DRY if is_dry else N_SIMS_FULL
    data_root = Path(args.data_root)
    data_dir  = data_root / "train"
    manifest_path = data_root / "manifest_train.json"

    run_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = run_dir / f"ae_{run_name}_{date_str}.log"
    log_fh   = open(log_path, "w")
    _real_stdout = sys.stdout
    _real_stderr = sys.stderr
    sys.stdout = Tee(log_fh)
    sys.stderr = Tee(log_fh)

    def log(msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"Run: {run_name}  ({'dry' if is_dry else 'full'})")
    log(f"Device: {DEVICE}  latent_dim: {LATENT_DIM}")
    log(f"Phase 1: {PHASE1_EPOCHS} epochs, Phase 2: {PHASE2_EPOCHS} epochs, lr: {LR}")

    manifest = load_manifest(manifest_path)
    stats    = load_stats(STATS_PATH)
    index    = manifest["index"][:n_sims]

    # ── Phase 1: full encoder + decoder ──────────────────────────────────────
    log("Phase 1: full 28-ch encoder + decoder (waveform reconstruction)...")

    full_dataset = CVDataset(str(data_dir), index, stats)
    full_loader  = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    enc1 = AutoencoderEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    dec  = WaveformDecoder(latent_dim=LATENT_DIM).to(DEVICE)
    log(f"  encoder params: {sum(p.numel() for p in enc1.parameters()):,}")
    log(f"  decoder params: {sum(p.numel() for p in dec.parameters()):,}")

    opt1 = torch.optim.Adam(list(enc1.parameters()) + list(dec.parameters()), lr=LR)
    mse  = nn.MSELoss()

    for epoch in range(1, PHASE1_EPOCHS + 1):
        enc1.train(); dec.train()
        total_loss = 0.0; n_batches = 0
        for _, x_full in full_loader:
            x_full = x_full.to(DEVICE)
            loss   = mse(dec(enc1(x_full)), x_full)
            opt1.zero_grad(); loss.backward(); opt1.step()
            total_loss += loss.item(); n_batches += 1
        log(f"  epoch {epoch:2d}/{PHASE1_EPOCHS}  mse={total_loss / n_batches:.4f}")

    torch.save(enc1.state_dict(), run_dir / "phase1_encoder.pt")
    torch.save(dec.state_dict(),  run_dir / "phase1_decoder.pt")
    log("Saved phase1_encoder.pt, phase1_decoder.pt")
    full_dataset.close()

    # ── Phase 2: reduced encoder, frozen decoder ──────────────────────────────
    log("Phase 2: reduced encoder (4-ch + scalars), frozen decoder...")

    paired_dataset = PairedCVDataset(str(data_dir), index, stats)
    paired_loader  = DataLoader(paired_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    enc2 = ReducedAutoencoderEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    dec.requires_grad_(False)
    log(f"  reduced encoder params: {sum(p.numel() for p in enc2.parameters()):,}")

    opt2 = torch.optim.Adam(enc2.parameters(), lr=LR)

    for epoch in range(1, PHASE2_EPOCHS + 1):
        enc2.train(); dec.eval()
        total_loss = 0.0; n_batches = 0
        for _, x_full, x_reduced in paired_loader:
            x_full    = x_full.to(DEVICE)
            x_reduced = x_reduced.to(DEVICE)
            loss      = mse(dec(enc2(x_reduced)), x_full)
            opt2.zero_grad(); loss.backward(); opt2.step()
            total_loss += loss.item(); n_batches += 1
        log(f"  epoch {epoch:2d}/{PHASE2_EPOCHS}  mse={total_loss / n_batches:.4f}")

    torch.save(enc2.state_dict(), run_dir / "phase2_encoder.pt")
    log("Saved phase2_encoder.pt")

    ae_info = dict(
        run=run_name,
        type=run_type,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        git_hash=git_hash(),
        device=DEVICE,
        data=dict(n_sims=n_sims, data_dir=str(data_dir)),
        model=dict(
            encoder="AutoencoderEncoder",
            reduced_encoder="ReducedAutoencoderEncoder",
            decoder="WaveformDecoder",
            latent_dim=LATENT_DIM,
        ),
        training=dict(
            phase1_epochs=PHASE1_EPOCHS,
            phase2_epochs=PHASE2_EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
        ),
    )
    (run_dir / "ae_info.json").write_text(json.dumps(ae_info, indent=2))
    log("Saved ae_info.json")

    paired_dataset.close()
    log_fh.close()
    sys.stdout = _real_stdout
    sys.stderr = _real_stderr
    print(f"\nLog written to {log_path}")


if __name__ == "__main__":
    main()
