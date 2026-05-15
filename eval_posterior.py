"""
Posterior evaluation — run interactively section by section.
Each # %% block is a cell you can send to the REPL in Zed.
"""

# %%
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from dataset import CVDataset, PARAM_KEYS, load_stats
from train_sbi import WaveformEmbedding  # needed to unpickle posterior

# %%  ── Config ──────────────────────────────────────────────────────────────

POSTERIOR_PATH = ROOT / 'outputs/exp_baseline_cnn4e64_maf5/posterior.pt'
STATS_PATH     = ROOT / 'norm_stats.json'
DATA_DIR       = Path('/media/8TBNVME/data/neh10/hdf5/cv8/simset_10M_cv8Eed_20260314/test')
MANIFEST       = DATA_DIR.parent / 'manifest_test.json'
TRAIN_DIR      = Path('/media/8TBNVME/data/neh10/hdf5/cv8/simset_10M_cv8Eed_20260314/train')
TRAIN_MANIFEST = DATA_DIR.parent / 'manifest_train.json'

BATCH_SIZE          = 256
BATCH_IDX           = 0
N_POSTERIOR_SAMPLES = 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# %%  ── Load posterior ─────────────────────────────────────────────────────

posterior = torch.load(POSTERIOR_PATH, map_location=device, weights_only=False)
print('Posterior loaded:', type(posterior).__name__)

# %%  ── Load test data ─────────────────────────────────────────────────────

stats    = load_stats(STATS_PATH)
manifest = json.load(open(MANIFEST))

start   = BATCH_IDX * BATCH_SIZE
entries = manifest['index'][start:start + BATCH_SIZE]
print(f'Test batch {BATCH_IDX}: sims {start}–{start + BATCH_SIZE - 1}')

ds = CVDataset(str(DATA_DIR), entries, stats)
loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0)
all_theta, all_x = [], []
for theta_b, x_b in loader:
    all_theta.append(theta_b)
    all_x.append(x_b)
ds.close()

theta_gt = torch.cat(all_theta)   # (256, 25) physical units
x_test   = torch.cat(all_x)       # (256, 5628)
print(f'theta_gt: {theta_gt.shape}  x_test: {x_test.shape}')

# %%  ── Diagnostic: embedding output variance ──────────────────────────────

emb_net = posterior.posterior_estimator.embedding_net

with torch.no_grad():
    emb_out = emb_net(x_test[:64].to(device))
print(f'Embedding output shape: {emb_out.shape}')
print(f'Per-dim std  min/mean/max: '
      f'{emb_out.std(0).min():.4f} / {emb_out.std(0).mean():.4f} / {emb_out.std(0).max():.4f}')
print(f'Overall mean: {emb_out.mean():.4f}  std: {emb_out.std():.4f}')

# %%  ── Diagnostic: sanity check on one TRAINING observation ───────────────

train_manifest = json.load(open(TRAIN_MANIFEST))
ds_tr = CVDataset(str(TRAIN_DIR), train_manifest['index'][:1], stats)
theta_tr, x_tr = ds_tr[0]
ds_tr.close()

with torch.no_grad():
    samples_tr = posterior.sample((500,), x=x_tr.to(device), show_progress_bars=False)
post_mean_tr = samples_tr.mean(0).cpu()

print(f"\nTraining obs sanity check:")
print(f"  {'param':<14} {'GT':>10} {'post mean':>12} {'err%':>8}")
for i, name in enumerate(PARAM_KEYS):
    err = abs(post_mean_tr[i].item() - theta_tr[i].item()) / (abs(theta_tr[i].item()) + 1e-9) * 100
    print(f"  {name:<14} {theta_tr[i].item():>10.4f} {post_mean_tr[i].item():>12.4f} {err:>7.1f}%")

# %%  ── Sample posterior on test set ───────────────────────────────────────

all_means, all_q05, all_q95 = [], [], []
for i, x_o in enumerate(x_test):
    samples = posterior.sample((N_POSTERIOR_SAMPLES,), x=x_o.to(device), show_progress_bars=False)
    all_means.append(samples.mean(dim=0).cpu())
    all_q05.append(samples.quantile(0.05, dim=0).cpu())
    all_q95.append(samples.quantile(0.95, dim=0).cpu())
    if (i + 1) % 50 == 0:
        print(f'  {i + 1}/{len(x_test)} done')

pred_mean = torch.stack(all_means).numpy()
q05       = torch.stack(all_q05).numpy()
q95       = torch.stack(all_q95).numpy()
gt_phys   = theta_gt.numpy()
print(f'pred_mean: {pred_mean.shape}')

# %%  ── Point-estimate metrics ──────────────────────────────────────────────

abs_err = np.abs(pred_mean - gt_phys)
ch_mae  = abs_err.mean(axis=0)
ch_med  = np.median(abs_err, axis=0)
ch_mape = (abs_err / (np.abs(gt_phys) + 1e-9) * 100).mean(axis=0)

print(f"{'Parameter':<14} {'MAE':>12} {'Median AE':>12} {'MAPE (%)':>10}")
print('-' * 52)
for i, name in enumerate(PARAM_KEYS):
    print(f'{name:<14} {ch_mae[i]:>12.4f} {ch_med[i]:>12.4f} {ch_mape[i]:>10.2f}%')
print(f'\nOverall  MAE mean: {ch_mae.mean():.4f}  MAPE: {ch_mape.mean():.2f}%')

# %%  ── Error bar charts ────────────────────────────────────────────────────

x = np.arange(len(PARAM_KEYS))
fig, axes = plt.subplots(1, 3, figsize=(22, 5))
axes[0].bar(x, ch_mae,  color='steelblue', alpha=0.85)
axes[1].bar(x, ch_med,  color='darkcyan',  alpha=0.85)
axes[2].bar(x, ch_mape, color='tomato',    alpha=0.85)
for ax, title, ylabel in zip(axes,
    ['Mean MAE', 'Median AE', 'MAPE (%)'],
    ['AE (physical units)', 'AE (physical units)', 'MAPE (%)']):
    ax.set_xticks(x); ax.set_xticklabels(PARAM_KEYS, rotation=45, ha='right')
    ax.set_title(title); ax.set_ylabel(ylabel)
fig.suptitle('SBI posterior mean errors — exp_baseline_cnn4e64_maf5', fontsize=13)
plt.tight_layout(); plt.show()

# %%  ── Scatter plots ───────────────────────────────────────────────────────

fig, axes = plt.subplots(5, 5, figsize=(20, 20))
for i, name in enumerate(PARAM_KEYS):
    ax = axes[i // 5, i % 5]
    gt_i, pred_i = gt_phys[:, i], pred_mean[:, i]
    lo, hi = gt_i.min(), gt_i.max()
    ax.scatter(gt_i, pred_i, s=10, alpha=0.5, color='steelblue')
    ax.plot([lo, hi], [lo, hi], color='tomato', lw=1, linestyle='--')
    ax.set_title(f'{name}  MAPE {ch_mape[i]:.1f}%', fontsize=9)
    ax.set_xlabel('GT', fontsize=8); ax.set_ylabel('Post. mean', fontsize=8)
    ax.tick_params(labelsize=7)
fig.suptitle('SBI posterior mean vs GT — exp_baseline_cnn4e64_maf5', fontsize=13)
plt.tight_layout(); plt.show()

# %%  ── 90% credible interval coverage ────────────────────────────────────

in_ci    = (gt_phys >= q05) & (gt_phys <= q95)
coverage = in_ci.mean(axis=0) * 100

print(f"{'Parameter':<14} {'90% CI coverage':>18}")
print('-' * 35)
for i, name in enumerate(PARAM_KEYS):
    flag = '  ✓' if 85 <= coverage[i] <= 95 else '  ←'
    print(f'{name:<14} {coverage[i]:>16.1f}%{flag}')
print(f'\nMean coverage: {coverage.mean():.1f}%  (target: 90%)')

x = np.arange(len(PARAM_KEYS))
fig, ax = plt.subplots(figsize=(14, 4))
colors = ['steelblue' if 85 <= c <= 95 else 'tomato' for c in coverage]
ax.bar(x, coverage, color=colors, alpha=0.85)
ax.axhline(90, color='black', linestyle='--', lw=1, label='target 90%')
ax.axhline(85, color='gray',  linestyle=':',  lw=0.8)
ax.axhline(95, color='gray',  linestyle=':',  lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(PARAM_KEYS, rotation=45, ha='right')
ax.set_ylabel('Coverage (%)'); ax.set_title('90% CI coverage — exp_baseline_cnn4e64_maf5')
ax.legend(); plt.tight_layout(); plt.show()
