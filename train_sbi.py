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
import torch.multiprocessing
import torch.nn as nn
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy('file_system')
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
from models import (
    WaveformEmbedding, ReducedWaveformEmbedding, TransformerWaveformEmbedding,
    ReducedAutoencoderEncoder, LipschitzReducedAutoencoderEncoder,
    EMBED_DIM, LATENT_DIM,
)


# ─── Config ──────────────────────────────────────────────────────────────────

STATS_PATH  = Path("norm_stats.json")

N_SIMS_FULL    = 100_000
N_SIMS_DRYRUN  = 512

BATCH_SIZE      = 512
HIDDEN_FEATURES = 128
NUM_TRANSFORMS  = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    parser.add_argument("--embedding",
                        choices=["cnn", "cnn-meanpool", "sumstats", "cnn-reduced",
                                 "transformer-reduced", "ae-reduced", "ae-reduced-lipschitz"],
                        default="cnn",
                        help="cnn: full 28-ch CNN attn pool; cnn-meanpool: full 28-ch CNN mean pool; "
                             "cnn-reduced: 4-ch CNN + scalar prefix tokens; "
                             "transformer-reduced: transformer encoder on 4-ch + scalars; "
                             "ae-reduced: load phase2_encoder.pt (ReducedAutoencoderEncoder, 128-dim); "
                             "ae-reduced-lipschitz: same with soft spectral-norm ceiling (LipschitzReducedAutoencoderEncoder); "
                             "sumstats: hand-crafted")
    parser.add_argument("--freeze-epochs", type=int, default=0,
                        help="Epochs to train embedding only before joint training (0 = no freeze)")
    parser.add_argument("--freeze-embedding", action="store_true",
                        help="Freeze embedding net throughout — only the flow is trained. "
                             "Use with --encoder-ckpt to train NSF on a fixed MMD-adapted encoder.")
    parser.add_argument("--encoder-ckpt", type=str, default=None,
                        help="Custom encoder checkpoint for ae-reduced (overrides run_dir/phase2_encoder.pt). "
                             "e.g. path to mmd_encoder.pt from train_mmd.py")
    parser.add_argument("--sn-ceiling", type=float, default=2.0,
                        help="Soft spectral-norm ceiling for ae-reduced-lipschitz (default: 2.0)")
    parser.add_argument("--flow-model", choices=["maf", "nsf"], default="maf",
                        help="Normalising flow architecture (default: maf)")
    parser.add_argument("--num-transforms", type=int, default=None,
                        help="Number of flow transforms (default: 5 for MAF, 8 for NSF)")
    parser.add_argument("--hidden-features", type=int, default=None,
                        help="Flow hidden features (default: 128 for MAF, 256 for NSF)")
    parser.add_argument("--n-sims", type=int, default=None,
                        help="Number of simulations to use (default: 512 for dry, 100000 for full)")
    args = parser.parse_args()

    # Flow defaults: MAF→5/128, NSF→8/256
    if args.num_transforms is None:
        args.num_transforms = 8 if args.flow_model == "nsf" else NUM_TRANSFORMS
    if args.hidden_features is None:
        args.hidden_features = 256 if args.flow_model == "nsf" else HIDDEN_FEATURES

    run_type, run_name, run_dir = parse_run(args.run)
    is_dry          = run_type == "dry"
    use_reduced              = args.embedding == "cnn-reduced"
    use_transformer          = args.embedding == "transformer-reduced"
    use_ae_reduced           = args.embedding in ("ae-reduced", "ae-reduced-lipschitz")
    use_ae_reduced_lipschitz = args.embedding == "ae-reduced-lipschitz"
    use_sumstats             = args.embedding == "sumstats"
    use_meanpool             = args.embedding == "cnn-meanpool"
    data_root   = Path(args.data_root)
    data_dir    = data_root / "train"
    manifest_path = data_root / "manifest_train.json"
    if is_dry:
        n_sims = N_SIMS_DRYRUN
    elif args.n_sims is not None:
        n_sims = args.n_sims
    else:
        n_sims = N_SIMS_FULL

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

    if use_ae_reduced:
        ckpt = Path(args.encoder_ckpt) if args.encoder_ckpt else run_dir / "phase2_encoder.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"{ckpt} not found.")
        if use_ae_reduced_lipschitz:
            enc = LipschitzReducedAutoencoderEncoder(latent_dim=LATENT_DIM, sn_ceiling=args.sn_ceiling)
        else:
            enc = ReducedAutoencoderEncoder(latent_dim=LATENT_DIM)
        enc.load_state_dict(torch.load(ckpt, map_location="cpu"))
        embedding_net  = enc
        embedding_info = enc.describe()
        embedding_info["ckpt"] = str(ckpt)
        z_score_x      = "none"
        dataset_cls    = ReducedCVDataset
        param_keys     = PARAM_KEYS_INFER
    elif use_transformer:
        embedding_net = TransformerWaveformEmbedding()
        embedding_info = embedding_net.describe()
        z_score_x   = "none"
        dataset_cls = ReducedCVDataset
        param_keys  = PARAM_KEYS_INFER
    elif use_reduced:
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
        command=" ".join(sys.argv),
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
            model=args.flow_model,
            hidden_features=args.hidden_features,
            num_transforms=args.num_transforms,
            z_score_theta="independent",
            z_score_x=z_score_x,
        ),
        training=dict(
            batch_size=BATCH_SIZE,
            freeze_epochs=args.freeze_epochs,
            freeze_embedding=args.freeze_embedding,
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
        model=args.flow_model,
        embedding_net=embedding_net,
        hidden_features=args.hidden_features,
        num_transforms=args.num_transforms,
        z_score_theta="independent",
        z_score_x=z_score_x,
    )

    inference = NPE(prior=prior, density_estimator=density_estimator_fn, device=DEVICE)
    inference.append_simulations(theta, x, data_device='cpu')

    if args.freeze_embedding:
        for p in embedding_net.parameters():
            p.requires_grad_(False)
        log("Embedding frozen before training — only flow parameters will be updated...")

    elif args.freeze_epochs > 0:
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
        # parametrize blocks torch.save — bake constrained weights into tensors before pickling
        if use_ae_reduced_lipschitz:
            import torch.nn.utils.parametrize as P
            for mod in posterior.posterior_estimator.embedding_net.modules():
                if P.is_parametrized(mod):
                    P.remove_parametrizations(mod, 'weight', leave_parametrized=True)
            log("Spectral-norm parametrizations removed (weights baked in) for serialization")
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
