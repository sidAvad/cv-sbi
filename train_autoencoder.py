"""
Phases 1 and 2 of autoencoder pre-training for cv-sbi.

Phase 1: AutoencoderEncoder (full 28-ch CNN) + WaveformDecoder trained to
         reconstruct all 28 waveforms from a 128-dim latent (MSE loss).
         Trains to convergence via early stopping on val MSE.
         Saves: phase1_encoder.pt, phase1_decoder.pt

Phase 2: Load phase1_decoder.pt and freeze it. Train ReducedAutoencoderEncoder
         (4-ch pressure CNN + 5 scalar prefix tokens → 128-dim) to reconstruct
         full waveforms through the frozen decoder (MSE loss, early stopping).
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
import torch.multiprocessing
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

torch.multiprocessing.set_sharing_strategy('file_system')

from dataset import (
    CVDataset, PairedCVDataset,
    load_stats, load_manifest,
)
from models import (
    AutoencoderEncoder, ReducedAutoencoderEncoder, LipschitzReducedAutoencoderEncoder,
    WaveformDecoder, LATENT_DIM,
)


STATS_PATH        = Path("norm_stats.json")
N_SIMS_DEFAULT    = 100_000
N_SIMS_DRY        = 512
BATCH_SIZE        = 512
PHASE1_MAX_EPOCHS = 150
PHASE2_MAX_EPOCHS = 100
PATIENCE          = 15
VAL_FRAC          = 0.1
LR                = 1e-3
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"


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


def load_full_data(data_dir, index, stats, log):
    dataset = CVDataset(str(data_dir), index, stats)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    xs, n, loaded = [], len(index), 0
    for _, x_batch in loader:
        xs.append(x_batch)
        loaded += len(x_batch)
        print(f"\r  loaded {loaded}/{n}", end="", flush=True)
    print()
    dataset.close()
    x_full = torch.cat(xs)
    log(f"Loaded {loaded} sims  x_full={tuple(x_full.shape[1:])}")
    return x_full


def load_paired_data(data_dir, index, stats, log):
    dataset = PairedCVDataset(str(data_dir), index, stats)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    x_fulls, x_reds, n, loaded = [], [], len(index), 0
    for _, x_full_batch, x_red_batch in loader:
        x_fulls.append(x_full_batch)
        x_reds.append(x_red_batch)
        loaded += len(x_full_batch)
        print(f"\r  loaded {loaded}/{n}", end="", flush=True)
    print()
    dataset.close()
    x_full = torch.cat(x_fulls)
    x_red  = torch.cat(x_reds)
    log(f"Loaded {loaded} paired sims  x_full={tuple(x_full.shape[1:])}  x_reduced={tuple(x_red.shape[1:])}")
    return x_full, x_red


def make_loaders(tensor_ds):
    n_val   = max(1, int(len(tensor_ds) * VAL_FRAC))
    n_train = len(tensor_ds) - n_val
    train_ds, val_ds = random_split(tensor_ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",       required=True,
                        help="e.g. exp_cnn4e64-ae-reduced_maf5_freeze-maf")
    parser.add_argument("--data-root", required=True,
                        help="Dataset root containing train/ and manifest_train.json")
    parser.add_argument("--n-sims", type=int, default=None,
                        help="Number of simulations (dry runs always 512; full runs default 100k)")
    parser.add_argument("--lipschitz", action="store_true",
                        help="Use LipschitzReducedAutoencoderEncoder in phase 2 (soft spectral-norm ceiling)")
    parser.add_argument("--sn-ceiling", type=float, default=2.0,
                        help="Soft spectral-norm ceiling per layer (default: 2.0, only with --lipschitz)")
    args = parser.parse_args()

    run_type, run_name, run_dir = parse_run(args.run)
    is_dry    = run_type == "dry"
    n_sims    = N_SIMS_DRY if is_dry else (args.n_sims or N_SIMS_DEFAULT)
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
    log(f"Phase 1: max {PHASE1_MAX_EPOCHS} epochs, Phase 2: max {PHASE2_MAX_EPOCHS} epochs")
    log(f"lr: {LR}  patience: {PATIENCE}  val_frac: {VAL_FRAC}")

    manifest = load_manifest(manifest_path)
    stats    = load_stats(STATS_PATH)
    index    = manifest["index"][:n_sims]

    # ── Phase 1: full encoder + decoder ──────────────────────────────────────
    log("Phase 1: loading full data into RAM...")
    x_full_all = load_full_data(data_dir, index, stats, log)
    train1, val1 = make_loaders(TensorDataset(x_full_all))

    enc1 = AutoencoderEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    dec  = WaveformDecoder(latent_dim=LATENT_DIM).to(DEVICE)
    log(f"  encoder params: {sum(p.numel() for p in enc1.parameters()):,}")
    log(f"  decoder params: {sum(p.numel() for p in dec.parameters()):,}")

    opt1 = torch.optim.Adam(list(enc1.parameters()) + list(dec.parameters()), lr=LR)
    mse  = nn.MSELoss()

    best_val1, wait1, phase1_epochs_run = float("inf"), 0, 0
    for epoch in range(1, PHASE1_MAX_EPOCHS + 1):
        enc1.train(); dec.train()
        train_loss = 0.0; n_train = 0
        for (x_batch,) in train1:
            x_batch = x_batch.to(DEVICE)
            loss    = mse(dec(enc1(x_batch)), x_batch)
            opt1.zero_grad(); loss.backward(); opt1.step()
            train_loss += loss.item() * len(x_batch); n_train += len(x_batch)

        enc1.eval(); dec.eval()
        val_loss = 0.0; n_val = 0
        with torch.no_grad():
            for (x_batch,) in val1:
                x_batch   = x_batch.to(DEVICE)
                val_loss  += mse(dec(enc1(x_batch)), x_batch).item() * len(x_batch)
                n_val     += len(x_batch)

        train_mse = train_loss / n_train
        val_mse   = val_loss   / n_val
        phase1_epochs_run = epoch
        log(f"  epoch {epoch:3d}/{PHASE1_MAX_EPOCHS}  train={train_mse:.4f}  val={val_mse:.4f}")

        if val_mse < best_val1:
            best_val1 = val_mse
            wait1     = 0
            torch.save(enc1.state_dict(), run_dir / "phase1_encoder.pt")
            torch.save(dec.state_dict(),  run_dir / "phase1_decoder.pt")
        else:
            wait1 += 1
            if wait1 >= PATIENCE:
                log(f"  early stop  best_val={best_val1:.4f}")
                break

    log(f"Phase 1 done: {phase1_epochs_run} epochs  best_val={best_val1:.4f}")
    log("Saved phase1_encoder.pt, phase1_decoder.pt")

    # reload best checkpoint into memory before phase 2 uses dec
    dec.load_state_dict(torch.load(run_dir / "phase1_decoder.pt", map_location=DEVICE))

    # ── Phase 2: reduced encoder, frozen decoder ──────────────────────────────
    log("Phase 2: loading paired data into RAM...")
    x_full_p2, x_red_p2 = load_paired_data(data_dir, index, stats, log)
    train2, val2 = make_loaders(TensorDataset(x_full_p2, x_red_p2))

    if args.lipschitz:
        enc2 = LipschitzReducedAutoencoderEncoder(latent_dim=LATENT_DIM, sn_ceiling=args.sn_ceiling).to(DEVICE)
        log(f"  using LipschitzReducedAutoencoderEncoder  sn_ceiling={args.sn_ceiling}")
    else:
        enc2 = ReducedAutoencoderEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    dec.requires_grad_(False)
    log(f"  reduced encoder params: {sum(p.numel() for p in enc2.parameters()):,}")

    opt2 = torch.optim.Adam(enc2.parameters(), lr=LR)

    best_val2, wait2, phase2_epochs_run = float("inf"), 0, 0
    for epoch in range(1, PHASE2_MAX_EPOCHS + 1):
        enc2.train(); dec.eval()
        train_loss = 0.0; n_train = 0
        for x_full_b, x_red_b in train2:
            x_full_b  = x_full_b.to(DEVICE)
            x_red_b   = x_red_b.to(DEVICE)
            loss      = mse(dec(enc2(x_red_b)), x_full_b)
            opt2.zero_grad(); loss.backward(); opt2.step()
            train_loss += loss.item() * len(x_full_b); n_train += len(x_full_b)

        enc2.eval()
        val_loss = 0.0; n_val = 0
        with torch.no_grad():
            for x_full_b, x_red_b in val2:
                x_full_b = x_full_b.to(DEVICE)
                x_red_b  = x_red_b.to(DEVICE)
                val_loss += mse(dec(enc2(x_red_b)), x_full_b).item() * len(x_full_b)
                n_val    += len(x_full_b)

        train_mse = train_loss / n_train
        val_mse   = val_loss   / n_val
        phase2_epochs_run = epoch
        sn_str = ""
        if args.lipschitz and epoch % 10 == 0:
            sn = enc2.spectral_norms()
            sn_max = max(sn.values())
            sn_str = f"  sn_max={sn_max:.3f}"
        log(f"  epoch {epoch:3d}/{PHASE2_MAX_EPOCHS}  train={train_mse:.4f}  val={val_mse:.4f}{sn_str}")

        if val_mse < best_val2:
            best_val2 = val_mse
            wait2     = 0
            torch.save(enc2.state_dict(), run_dir / "phase2_encoder.pt")
        else:
            wait2 += 1
            if wait2 >= PATIENCE:
                log(f"  early stop  best_val={best_val2:.4f}")
                break

    log(f"Phase 2 done: {phase2_epochs_run} epochs  best_val={best_val2:.4f}")
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
            reduced_encoder="LipschitzReducedAutoencoderEncoder" if args.lipschitz else "ReducedAutoencoderEncoder",
            decoder="WaveformDecoder",
            latent_dim=LATENT_DIM,
            sn_ceiling=args.sn_ceiling if args.lipschitz else None,
        ),
        training=dict(
            phase1_epochs_run=phase1_epochs_run,
            phase1_best_val=best_val1,
            phase2_epochs_run=phase2_epochs_run,
            phase2_best_val=best_val2,
            phase1_max_epochs=PHASE1_MAX_EPOCHS,
            phase2_max_epochs=PHASE2_MAX_EPOCHS,
            patience=PATIENCE,
            val_frac=VAL_FRAC,
            batch_size=BATCH_SIZE,
            lr=LR,
        ),
    )
    (run_dir / "ae_info.json").write_text(json.dumps(ae_info, indent=2))
    log("Saved ae_info.json")

    log_fh.close()
    sys.stdout = _real_stdout
    sys.stderr = _real_stderr
    print(f"\nLog written to {log_path}")


if __name__ == "__main__":
    main()
