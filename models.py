"""
All embedding and autoencoder architectures for cv-sbi.

SBI embeddings (used by train_sbi.py):
  WaveformEmbedding            : full 28-ch CNN → embed_dim (64)
  ReducedWaveformEmbedding     : 4-ch CNN + scalar prefix tokens → embed_dim (64)
  TransformerWaveformEmbedding : transformer encoder on 4-ch + scalars → embed_dim (64)

AE pre-training components (used by train_autoencoder.py + train_sbi.py ae-reduced):
  AutoencoderEncoder           : full 28-ch CNN → latent_dim (128), phase 1
  ReducedAutoencoderEncoder    : 4-ch CNN + scalar prefix tokens → latent_dim (128), phase 2
  WaveformDecoder              : latent_dim → 512 → N_CHANNELS*T, phases 1 + 2
"""

import torch
import torch.nn as nn

from dataset import N_CHANNELS, N_REDUCED_CHANNELS, N_SCALARS, T

EMBED_DIM  = 64
LATENT_DIM = 128


# ── SBI embedding nets ────────────────────────────────────────────────────────

class WaveformEmbedding(nn.Module):
    """Full 28-ch 1D CNN → embed_dim. Pooling: 'attention' or 'mean'."""

    CONV_LAYERS = [
        (N_CHANNELS, 64,  7, 1),
        (64,         128, 5, 2),
        (128,        256, 5, 2),
        (256,        256, 3, 1),
    ]

    def __init__(self, embed_dim=EMBED_DIM, pooling="attention"):
        super().__init__()
        self.n_channels = N_CHANNELS
        self.t          = T
        self.embed_dim  = embed_dim
        self.pooling    = pooling

        layers = []
        for in_ch, out_ch, k, s in self.CONV_LAYERS:
            layers += [nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2, stride=s), nn.SiLU()]
        self.cnn = nn.Sequential(*layers)
        if pooling == "attention":
            self.attn_pool = nn.Linear(self.CONV_LAYERS[-1][1], 1)
        self.proj = nn.Linear(self.CONV_LAYERS[-1][1], embed_dim)

    def describe(self):
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

    def forward(self, x):
        x = x.view(-1, self.n_channels, self.t)
        h = self.cnn(x).transpose(1, 2)                          # (B, T', 256)
        if self.pooling == "attention":
            w = self.attn_pool(h).softmax(dim=1)
            h = (w * h).sum(dim=1)
        else:
            h = h.mean(dim=1)
        return self.proj(h)


class ReducedWaveformEmbedding(nn.Module):
    """4-ch pressure CNN + 5 scalars as prefix tokens → attention pool → embed_dim."""

    CONV_LAYERS = [
        (N_REDUCED_CHANNELS, 64,  7, 1),
        (64,                 128, 5, 2),
        (128,                256, 5, 2),
        (256,                256, 3, 1),
    ]

    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        self.wave_len  = N_REDUCED_CHANNELS * T  # 804
        feat_dim = self.CONV_LAYERS[-1][1]

        layers = []
        for in_ch, out_ch, k, s in self.CONV_LAYERS:
            layers += [nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2, stride=s), nn.SiLU()]
        self.cnn          = nn.Sequential(*layers)
        self.scalar_projs = nn.ModuleList([nn.Linear(1, feat_dim) for _ in range(N_SCALARS)])
        self.attn_pool    = nn.Linear(feat_dim, 1)
        self.proj         = nn.Linear(feat_dim, embed_dim)

    @property
    def output_dim(self):
        return self.embed_dim

    def describe(self):
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

    def forward(self, x):
        waves   = x[:, :self.wave_len].view(-1, N_REDUCED_CHANNELS, T)
        scalars = x[:, self.wave_len:]                                     # (B, 5)

        h = self.cnn(waves).transpose(1, 2)                                # (B, T', 256)
        scalar_tokens = torch.stack(
            [proj(scalars[:, i:i+1]) for i, proj in enumerate(self.scalar_projs)],
            dim=1,
        )                                                                   # (B, 5, 256)
        h = torch.cat([scalar_tokens, h], dim=1)                           # (B, T'+5, 256)
        w = self.attn_pool(h).softmax(dim=1)
        h = (w * h).sum(dim=1)
        return self.proj(h)


class TransformerWaveformEmbedding(nn.Module):
    """Transformer encoder: 4 pressure waveforms + 5 scalar prefix tokens → embed_dim.

    201 timesteps → (B, T, 4) tokens + 5 scalar prefix tokens → (B, T+5, d_model)
    → transformer → attention pool → embed_dim.
    Learnable positional embeddings on waveform tokens only.
    """

    D_MODEL  = 64
    N_HEADS  = 4
    N_LAYERS = 3
    FFN_DIM  = 128

    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        self.wave_len  = N_REDUCED_CHANNELS * T  # 804

        self.input_proj   = nn.Linear(N_REDUCED_CHANNELS, self.D_MODEL)
        self.pos_emb      = nn.Embedding(T, self.D_MODEL)
        self.scalar_projs = nn.ModuleList(
            [nn.Linear(1, self.D_MODEL) for _ in range(N_SCALARS)]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.D_MODEL, nhead=self.N_HEADS,
            dim_feedforward=self.FFN_DIM, dropout=0.0,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.N_LAYERS)
        self.attn_pool   = nn.Linear(self.D_MODEL, 1)
        self.proj        = nn.Linear(self.D_MODEL, embed_dim)

    def describe(self):
        return {
            "type": "TransformerWaveformEmbedding",
            "input_waveforms": f"({N_REDUCED_CHANNELS}, {T})",
            "input_scalars": "Pas_mean, Pas_max, Pas_min, SV, HR_z",
            "d_model": self.D_MODEL,
            "n_heads": self.N_HEADS,
            "n_layers": self.N_LAYERS,
            "ffn_dim": self.FFN_DIM,
            "seq_len": T + N_SCALARS,
            "pooling": "attention",
            "embed_dim": self.embed_dim,
            "n_params": sum(p.numel() for p in self.parameters()),
        }

    def forward(self, x):
        waves   = x[:, :self.wave_len].view(-1, T, N_REDUCED_CHANNELS)    # (B, T, 4)
        scalars = x[:, self.wave_len:]                                      # (B, 5)

        pos = torch.arange(T, device=x.device)
        h   = self.input_proj(waves) + self.pos_emb(pos)                   # (B, T, D_MODEL)

        scalar_tokens = torch.stack(
            [proj(scalars[:, i:i+1]) for i, proj in enumerate(self.scalar_projs)],
            dim=1,
        )                                                                    # (B, 5, D_MODEL)
        h = torch.cat([scalar_tokens, h], dim=1)                            # (B, T+5, D_MODEL)
        h = self.transformer(h)

        w = self.attn_pool(h).softmax(dim=1)
        h = (w * h).sum(dim=1)
        return self.proj(h)


# ── Autoencoder components ────────────────────────────────────────────────────

class AutoencoderEncoder(nn.Module):
    """Full 28-ch CNN → latent_dim. Used in phase 1 of AE pre-training."""

    CONV_LAYERS = [
        (N_CHANNELS, 64,  7, 1),
        (64,         128, 5, 2),
        (128,        256, 5, 2),
        (256,        256, 3, 1),
    ]

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        feat_dim = self.CONV_LAYERS[-1][1]

        layers = []
        for in_ch, out_ch, k, s in self.CONV_LAYERS:
            layers += [nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2, stride=s), nn.SiLU()]
        self.cnn       = nn.Sequential(*layers)
        self.attn_pool = nn.Linear(feat_dim, 1)
        self.proj      = nn.Linear(feat_dim, latent_dim)

    def describe(self):
        return {
            "type": "AutoencoderEncoder",
            "input": f"({N_CHANNELS}, {T})",
            "latent_dim": self.latent_dim,
            "n_params": sum(p.numel() for p in self.parameters()),
        }

    def forward(self, x):
        h = self.cnn(x.view(-1, N_CHANNELS, T)).transpose(1, 2)  # (B, T', 256)
        w = self.attn_pool(h).softmax(dim=1)
        h = (w * h).sum(dim=1)
        return self.proj(h)


class ReducedAutoencoderEncoder(nn.Module):
    """4-ch pressure CNN + 5 scalar prefix tokens → latent_dim.

    Trained in phase 2 to reconstruct full waveforms via the frozen phase-1 decoder.
    Loaded as embedding_net in train_sbi.py --embedding ae-reduced (phases 3+4).
    """

    CONV_LAYERS = [
        (N_REDUCED_CHANNELS, 64,  7, 1),
        (64,                 128, 5, 2),
        (128,                256, 5, 2),
        (256,                256, 3, 1),
    ]

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.wave_len   = N_REDUCED_CHANNELS * T  # 804
        feat_dim = self.CONV_LAYERS[-1][1]

        layers = []
        for in_ch, out_ch, k, s in self.CONV_LAYERS:
            layers += [nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2, stride=s), nn.SiLU()]
        self.cnn          = nn.Sequential(*layers)
        self.scalar_projs = nn.ModuleList([nn.Linear(1, feat_dim) for _ in range(N_SCALARS)])
        self.attn_pool    = nn.Linear(feat_dim, 1)
        self.proj         = nn.Linear(feat_dim, latent_dim)

    @property
    def output_dim(self):
        return self.latent_dim

    def describe(self):
        return {
            "type": "ReducedAutoencoderEncoder",
            "input_waveforms": f"({N_REDUCED_CHANNELS}, {T})",
            "input_scalars": "Pas_mean, Pas_max, Pas_min, SV, HR_z",
            "latent_dim": self.latent_dim,
            "n_params": sum(p.numel() for p in self.parameters()),
        }

    def forward(self, x):
        waves   = x[:, :self.wave_len].view(-1, N_REDUCED_CHANNELS, T)
        scalars = x[:, self.wave_len:]                                    # (B, 5)

        h = self.cnn(waves).transpose(1, 2)                               # (B, T', 256)
        scalar_tokens = torch.stack(
            [proj(scalars[:, i:i+1]) for i, proj in enumerate(self.scalar_projs)],
            dim=1,
        )                                                                  # (B, 5, 256)
        h = torch.cat([scalar_tokens, h], dim=1)                          # (B, T'+5, 256)
        w = self.attn_pool(h).softmax(dim=1)
        h = (w * h).sum(dim=1)
        return self.proj(h)


class WaveformDecoder(nn.Module):
    """MLP decoder: latent_dim → 512 → N_CHANNELS*T. Used in phases 1 and 2."""

    def __init__(self, latent_dim=LATENT_DIM, hidden=512):
        super().__init__()
        out_dim = N_CHANNELS * T  # 5628
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def describe(self):
        return {
            "type": "WaveformDecoder",
            "latent_dim": self.net[0].in_features,
            "hidden": self.net[0].out_features,
            "output_dim": N_CHANNELS * T,
            "n_params": sum(p.numel() for p in self.parameters()),
        }

    def forward(self, z):
        return self.net(z)
