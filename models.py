"""
All embedding and autoencoder architectures for cv-sbi.

SBI embeddings (used by train_sbi.py):
  WaveformEmbedding            : full 28-ch CNN → embed_dim (64)
  ReducedWaveformEmbedding     : 4-ch CNN + scalar prefix tokens → embed_dim (64)
  TransformerWaveformEmbedding : transformer encoder on 4-ch + scalars → embed_dim (64)

AE pre-training components — legacy (old runs):
  AutoencoderEncoder                : full 28-ch CNN → latent_dim (128), phase 1
  ReducedAutoencoderEncoder         : 4-ch CNN + scalar prefix tokens → latent_dim (128), phase 2
  LipschitzReducedAutoencoderEncoder: same as above + soft spectral-norm ceiling per layer
  WaveformDecoder                   : latent_dim → 512 → N_CHANNELS*T, phases 1 + 2

AE pre-training components — v2 (new runs, hourglass encoder + multi-layer decoder):
  ContAutoencoderEncoder            : 24-ch hourglass CNN → latent_dim, phase 1
  LipschitzContAutoencoderEncoder   : same + soft spectral-norm ceiling per layer, phase 1
  MultiHeadWaveformDecoder          : latent_dim → deep MLP → (B, N_CONT, T), phases 1 + 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P

from dataset import N_CHANNELS, N_CONT, N_REDUCED_CHANNELS, N_SCALARS, T

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


class _SoftSpectralCeiling(nn.Module):
    """Weight parametrization: scale W down if sigma_1(W) exceeds `ceiling`, else leave unchanged.

    Uses one step of power iteration per forward pass, warm-started from the previous step,
    so the sigma_1 estimate is accurate after the first few training steps.

    scale = min(1, ceiling / sigma_1)  →  effective sigma_1 = min(sigma_1, ceiling)
    """

    def __init__(self, ceiling: float, n_power_iters: int = 1):
        super().__init__()
        self.ceiling       = ceiling
        self.n_power_iters = n_power_iters
        self.register_buffer('_u', None)
        self.register_buffer('_v', None)
        self.last_sigma: float = 0.0   # updated each forward; readable for logging

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        W_mat = W.reshape(W.shape[0], -1)            # (C_out, C_in*k) or (out, in)
        h, w  = W_mat.shape

        # warm-start or re-initialise if shape changed
        if self._u is None or self._u.shape[0] != h:
            self._u = F.normalize(W_mat.new_empty(h).normal_(), dim=0)
        if self._v is None or self._v.shape[0] != w:
            self._v = F.normalize(W_mat.new_empty(w).normal_(), dim=0)

        u, v = self._u.detach(), self._v.detach()
        with torch.no_grad():
            for _ in range(self.n_power_iters):
                v = F.normalize(W_mat.t() @ u, dim=0, eps=1e-12)
                u = F.normalize(W_mat @ v,     dim=0, eps=1e-12)

        sigma = (u @ (W_mat @ v)).abs()

        if self.training:
            self._u.copy_(u)
            self._v.copy_(v)

        # scale = 1 when sigma <= ceiling; ceiling/sigma when sigma > ceiling
        scale = (self.ceiling / sigma.clamp(min=1e-12)).clamp(max=1.0)
        # store effective sigma (= min(sigma, ceiling)) so spectral_norms() is directly checkable
        self.last_sigma = (sigma * scale).item()
        return W * scale

    def right_inverse(self, W: torch.Tensor) -> torch.Tensor:
        return W   # identity: parametrize stores the original weight as-is


def _sn(module: nn.Module, ceiling: float) -> nn.Module:
    P.register_parametrization(module, 'weight', _SoftSpectralCeiling(ceiling))
    return module


class LipschitzReducedAutoencoderEncoder(nn.Module):
    """ReducedAutoencoderEncoder with a soft spectral-norm ceiling on every weight matrix.

    Each Conv1d and Linear layer is wrapped so its spectral norm (largest singular value)
    is clamped to at most `sn_ceiling`.  When sigma_1 <= sn_ceiling the weights are
    unchanged; when sigma_1 > sn_ceiling the weight is rescaled to sigma_1 = sn_ceiling.

    This prevents the encoder from arbitrarily stretching latent space, which is the root
    cause of real patients landing far outside the sim distribution.  The decoder and flow
    are NOT constrained — only the encoder maps inputs to the shared latent space.

    `spectral_norms()` returns the last estimated sigma_1 per layer; call it after a
    forward pass to verify the ceiling is active during training.
    """

    CONV_LAYERS = ReducedAutoencoderEncoder.CONV_LAYERS

    def __init__(self, latent_dim: int = LATENT_DIM, sn_ceiling: float = 2.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.sn_ceiling = sn_ceiling
        self.wave_len   = N_REDUCED_CHANNELS * T
        feat_dim = self.CONV_LAYERS[-1][1]

        layers = []
        for in_ch, out_ch, k, s in self.CONV_LAYERS:
            layers += [
                _sn(nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2, stride=s), sn_ceiling),
                nn.SiLU(),
            ]
        self.cnn          = nn.Sequential(*layers)
        self.scalar_projs = nn.ModuleList(
            [_sn(nn.Linear(1, feat_dim), sn_ceiling) for _ in range(N_SCALARS)]
        )
        self.attn_pool    = _sn(nn.Linear(feat_dim, 1), sn_ceiling)
        self.proj         = _sn(nn.Linear(feat_dim, latent_dim), sn_ceiling)

    @property
    def output_dim(self):
        return self.latent_dim

    def spectral_norms(self) -> dict[str, float]:
        """Return last estimated sigma_1 for every constrained layer.

        Call after a forward pass (values are 0.0 before the first forward).
        All values should be <= sn_ceiling if the ceiling is active.
        """
        out = {}
        for name, mod in self.named_modules():
            if isinstance(mod, _SoftSpectralCeiling):
                out[name] = mod.last_sigma
        return out

    def describe(self):
        return {
            "type": "LipschitzReducedAutoencoderEncoder",
            "input_waveforms": f"({N_REDUCED_CHANNELS}, {T})",
            "input_scalars": "Pas_mean, Pas_max, Pas_min, SV, HR_z",
            "latent_dim": self.latent_dim,
            "sn_ceiling": self.sn_ceiling,
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


# ── AE v2: hourglass encoder + deep decoder (new runs) ───────────────────────

class ContAutoencoderEncoder(nn.Module):
    """24-ch continuous CNN → latent_dim. Hourglass channel structure.

    Phase 1 of AE v2 pre-training. Takes only the 24 z-scored continuous
    waveform channels (no valve signals). Expand-then-contract channel taper
    distributes bottleneck pressure through the conv hierarchy rather than
    dumping all compression onto the final linear projection.
    """

    CONV_LAYERS = [
        (N_CONT, 64,  7, 1),   # 24 → 64,  local features
        (64,     128, 5, 2),   # 64 → 128, expand, stride-2 downsample
        (128,    64,  5, 2),   # 128 → 64, contract, stride-2 downsample
        (64,     64,  3, 1),   # 64 → 64,  refine
    ]

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        feat_dim = self.CONV_LAYERS[-1][1]  # 64

        layers = []
        for in_ch, out_ch, k, s in self.CONV_LAYERS:
            layers += [nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2, stride=s), nn.SiLU()]
        self.cnn       = nn.Sequential(*layers)
        self.attn_pool = nn.Linear(feat_dim, 1)
        self.proj      = nn.Linear(feat_dim, latent_dim)

    def describe(self):
        return {
            "type": "ContAutoencoderEncoder",
            "input": f"({N_CONT}, {T})",
            "latent_dim": self.latent_dim,
            "n_params": sum(p.numel() for p in self.parameters()),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.cnn(x.view(-1, N_CONT, T)).transpose(1, 2)  # (B, T', 64)
        w = self.attn_pool(h).softmax(dim=1)
        h = (w * h).sum(dim=1)                                # (B, 64)
        return self.proj(h)                                   # (B, latent_dim)


class LipschitzContAutoencoderEncoder(nn.Module):
    """ContAutoencoderEncoder with a soft spectral-norm ceiling on every weight matrix.

    Phase 1 of AE v2 pre-training when --lipschitz is set. Prevents the encoder
    from arbitrarily stretching the latent space, keeping the sim distribution
    compact and amenable to OT transport.
    """

    CONV_LAYERS = ContAutoencoderEncoder.CONV_LAYERS

    def __init__(self, latent_dim: int = LATENT_DIM, sn_ceiling: float = 2.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.sn_ceiling = sn_ceiling
        feat_dim = self.CONV_LAYERS[-1][1]  # 64

        layers = []
        for in_ch, out_ch, k, s in self.CONV_LAYERS:
            layers += [
                _sn(nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2, stride=s), sn_ceiling),
                nn.SiLU(),
            ]
        self.cnn       = nn.Sequential(*layers)
        self.attn_pool = _sn(nn.Linear(feat_dim, 1), sn_ceiling)
        self.proj      = _sn(nn.Linear(feat_dim, latent_dim), sn_ceiling)

    @property
    def output_dim(self):
        return self.latent_dim

    def spectral_norms(self) -> dict[str, float]:
        out = {}
        for name, mod in self.named_modules():
            if isinstance(mod, _SoftSpectralCeiling):
                out[name] = mod.last_sigma
        return out

    def describe(self):
        return {
            "type": "LipschitzContAutoencoderEncoder",
            "input": f"({N_CONT}, {T})",
            "latent_dim": self.latent_dim,
            "sn_ceiling": self.sn_ceiling,
            "n_params": sum(p.numel() for p in self.parameters()),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.cnn(x.view(-1, N_CONT, T)).transpose(1, 2)  # (B, T', 64)
        w = self.attn_pool(h).softmax(dim=1)
        h = (w * h).sum(dim=1)                                # (B, 64)
        return self.proj(h)                                   # (B, latent_dim)


class MultiHeadWaveformDecoder(nn.Module):
    """Deep MLP decoder: latent_dim → trunk (n_layers × hidden) → (B, N_CONT, T).

    Replaces WaveformDecoder for AE v2 runs. Outputs only the 24 z-scored
    continuous channels — valve signals are dropped (they are redundant given
    the pressure/flow/volume channels). The deeper trunk gives the decoder
    capacity to invert the hourglass encoder.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, hidden: int = 1024, n_layers: int = 6):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden     = hidden
        self.n_layers   = n_layers

        trunk = [nn.Linear(latent_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            trunk += [nn.Linear(hidden, hidden), nn.SiLU()]
        self.trunk     = nn.Sequential(*trunk)
        self.cont_head = nn.Linear(hidden, N_CONT * T)

    def describe(self):
        return {
            "type": "MultiHeadWaveformDecoder",
            "latent_dim": self.latent_dim,
            "hidden": self.hidden,
            "n_layers": self.n_layers,
            "output": f"({N_CONT}, {T})",
            "n_params": sum(p.numel() for p in self.parameters()),
        }

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        return self.cont_head(self.trunk(z)).view(B, N_CONT, T)  # (B, N_CONT, T)
