"""
Phase 4 (OT): Adapt enc_reduced to real patients via Sinkhorn OT.

Trains a real-patient encoder (enc_real) to map real latents toward the sim
latent distribution defined by a frozen copy of the same checkpoint (enc_frozen).
Pure Sinkhorn divergence — no flow LL, no anchor loss.

At inference: z_real = real_encoder(x_real), z_sim = base posterior embedding_net(x_sim).
NN search compares z_real to z_sim; posterior conditions on x_nn via the base embedding_net.

Usage:
    python train_ot.py
        --run exp_cnn4e64-ae-reduced_maf5_freeze-maf_1M
        --real-data ~/real_data/multibeat
        --output-run exp_cnn4e64-ae-reduced_1M_ot-sinkhorn
        --sim-data-root /media/local/SimData/hdf5/cv8/simset_10M_cv8Eed_20260314
        [--encoder-ckpt path/to/enc_joint.pt]
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.multiprocessing
from geomloss import SamplesLoss
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy('file_system')

from dataset import ReducedCVDataset, load_stats, load_manifest
from models import ReducedAutoencoderEncoder, LATENT_DIM


STATS_PATH     = Path("norm_stats.json")
WAVE_KEYS_REAL = ["Prv", "Pra", "Pvp", "Pap"]
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


def load_real_patients(data_dir: Path, stats: dict, log) -> list[torch.Tensor]:
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

                beats.append(torch.cat([waves_t.reshape(-1), scalars]))

        if beats:
            patient_tensors.append(torch.stack(beats))

    n_patients = len(patient_tensors)
    n_beats    = sum(t.shape[0] for t in patient_tensors)
    log(f"Loaded {n_patients} patients, {n_beats} beats total from {data_dir.name}")
    return patient_tensors


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
                        help="Base ae-reduced run (provides posterior.pt)")
    parser.add_argument("--real-data",     required=True)
    parser.add_argument("--output-run",    required=True,
                        help="Output run name")
    parser.add_argument("--sim-data-root", required=True)
    parser.add_argument("--encoder-ckpt",  default=None,
                        help="Encoder checkpoint for both enc_real (trainable) and enc_frozen (sim target). "
                             "Default: phase2_encoder.pt from --run.")
    parser.add_argument("--latent-dim",     type=int,   default=LATENT_DIM,
                        help="Latent dimension of the AE encoder (must match checkpoint, default: 128)")
    parser.add_argument("--n-sim",         type=int,   default=N_SIM_DEFAULT)
    parser.add_argument("--sim-batch",     type=int,   default=512,
                        help="Sim latents subsampled per epoch from z_sim_all (precomputed once).")
    parser.add_argument("--lr",            type=float, default=LR)
    parser.add_argument("--weight-decay",  type=float, default=1e-4)
    parser.add_argument("--grad-clip",     type=float, default=1.0)
    parser.add_argument("--blur",          type=float, default=0.5,
                        help="Sinkhorn blur (entropic regularisation ε).")
    parser.add_argument("--epochs",        type=int,   default=MAX_EPOCHS)
    parser.add_argument("--patience",      type=int,   default=PATIENCE)
    parser.add_argument("--dry-run",       action="store_true",
                        help="Smoke test: 2 epochs, 512 sim samples, no posterior saved")
    args = parser.parse_args()

    if args.dry_run:
        args.epochs    = 2
        args.n_sim     = 512
        args.sim_batch = 512
        args.patience  = 999

    base_run_dir = Path("outputs") / args.run
    out_root     = Path("dry-runs") if args.dry_run else Path("outputs")
    ot_run_dir   = out_root / args.output_run
    ot_run_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = ot_run_dir / f"ot_{args.output_run}_{date_str}.log"
    log_fh   = open(log_path, "w")
    _stdout, _stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(log_fh)
    sys.stderr = Tee(log_fh)

    def log(msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"OT fine-tuning: {args.run} → {args.output_run}  ({'DRY RUN' if args.dry_run else 'full'})")
    log(f"Device: {DEVICE}  lr={args.lr}  blur={args.blur}")

    stats    = load_stats(STATS_PATH)
    manifest = load_manifest(Path(args.sim_data_root) / "manifest_train.json")

    patient_tensors = load_real_patients(Path(args.real_data), stats, log)

    log(f"Loading {args.n_sim} sim observations into RAM...")
    x_sim = load_sim_data(
        Path(args.sim_data_root) / "train", manifest, stats, args.n_sim, log
    )

    ckpt = Path(args.encoder_ckpt) if args.encoder_ckpt else base_run_dir / "phase2_encoder.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {ckpt}")
    log(f"Encoder checkpoint: {ckpt}")

    enc = ReducedAutoencoderEncoder(latent_dim=args.latent_dim).to(DEVICE)
    enc.load_state_dict(torch.load(ckpt, map_location=DEVICE))

    enc_frozen = ReducedAutoencoderEncoder(latent_dim=args.latent_dim).to(DEVICE)
    enc_frozen.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    enc_frozen.requires_grad_(False)
    enc_frozen.eval()

    log("Pre-computing sim latents (enc_frozen, all sims)...")
    z_sim_chunks = []
    with torch.no_grad():
        for i in range(0, len(x_sim), 512):
            z_sim_chunks.append(enc_frozen(x_sim[i:i+512].to(DEVICE)))
    z_sim_all = torch.cat(z_sim_chunks)
    del x_sim, z_sim_chunks
    log(f"z_sim_all: {tuple(z_sim_all.shape)}  GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    sinkhorn = SamplesLoss("sinkhorn", p=2, blur=args.blur, backend="tensorized")
    log(f"Sinkhorn: p=2  blur={args.blur}  backend=tensorized")

    with torch.no_grad():
        z_real_init = torch.stack([enc(beats.to(DEVICE)).mean(0) for beats in patient_tensors])
    log(f"Initial Sinkhorn: {sinkhorn(z_real_init, z_sim_all).item():.4f}")

    opt = torch.optim.Adam(enc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    log(f"Adam lr={args.lr}  weight_decay={args.weight_decay}  grad_clip={args.grad_clip}")

    best_loss  = float("inf")
    wait       = 0
    best_state = {k: v.clone() for k, v in enc.state_dict().items()}

    csv_path = ot_run_dir / f"train_log_{date_str}.csv"
    csv_fh   = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_fh)
    csv_writer.writerow(["epoch", "ot_loss"])

    log("Training...")
    for epoch in range(1, args.epochs + 1):
        enc.train()
        z_real_list = [enc(beats.to(DEVICE)).mean(dim=0) for beats in patient_tensors]
        z_real = torch.stack(z_real_list)

        idx  = torch.randperm(len(z_sim_all), device=DEVICE)[:args.sim_batch]
        loss = sinkhorn(z_real, z_sim_all[idx])

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), args.grad_clip)
        opt.step()

        loss_val = loss.item()
        csv_writer.writerow([epoch, f"{loss_val:.6f}"])
        csv_fh.flush()

        if loss_val < best_loss:
            best_loss  = loss_val
            wait       = 0
            best_state = {k: v.clone() for k, v in enc.state_dict().items()}
        else:
            wait += 1

        if epoch % LOG_EVERY == 0 or epoch == 1:
            log(f"  epoch {epoch:4d}/{args.epochs}  ot={loss_val:.4f}  best={best_loss:.4f}  wait={wait}")

        if wait >= args.patience:
            log(f"  early stop at epoch {epoch}  best={best_loss:.4f}")
            break

    csv_fh.close()

    enc.load_state_dict(best_state)
    torch.save(enc.state_dict(), ot_run_dir / "real_encoder.pt")
    log("Saved real_encoder.pt")

    if args.dry_run:
        log("Dry run — posterior not copied.")
    else:
        import shutil
        shutil.copy(base_run_dir / "posterior.pt", ot_run_dir / "posterior.pt")
        log(f"Copied posterior.pt from {base_run_dir.name}")

    run_info = dict(
        run=args.output_run,
        command=" ".join(sys.argv),
        base_run=args.run,
        encoder_ckpt=str(ckpt),
        real_data=str(args.real_data),
        n_patients=len(patient_tensors),
        n_beats=sum(t.shape[0] for t in patient_tensors),
        timestamp=datetime.now().isoformat(timespec="seconds"),
        git_hash=git_hash(),
        device=DEVICE,
        training=dict(
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            blur=args.blur,
            max_epochs=args.epochs,
            patience=args.patience,
            n_sim=args.n_sim,
            sim_batch=args.sim_batch,
            best_ot_loss=best_loss,
        ),
    )
    (ot_run_dir / f"run_info_{date_str}.json").write_text(json.dumps(run_info, indent=2))
    log("Saved run_info")

    log_fh.close()
    sys.stdout = _stdout
    sys.stderr = _stderr
    print(f"\nLog written to {log_path}")


if __name__ == "__main__":
    main()
