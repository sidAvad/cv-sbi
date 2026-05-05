"""
Flat decoder MLP surrogate for the cardiovascular ODE simulator.

Takes a 25-dim parameter vector and produces:
  - 24 continuous waveforms of length 201 (z-score normalised space)
  - 4 binary valve signals of length 201 (raw logits — apply sigmoid to read)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CVSurrogate(nn.Module):
    """
    Architecture
    ------------
    params (B, 25) → shared trunk (Linear → SiLU × n_layers, width=hidden)
                   → cont  head: Linear(hidden, n_cont  × T) → (B, n_cont,  T)
                   → valve head: Linear(hidden, n_valve × T) → (B, n_valve, T)

    The valve head emits raw logits (loss uses BCEWithLogitsLoss); apply a
    sigmoid at inference time to read them as probabilities.
    """

    def __init__(
        self,
        n_params: int = 25,
        n_cont: int = 24,
        n_valve: int = 4,
        T: int = 201,
        hidden: int = 1024,
        n_layers: int = 6,
    ):
        super().__init__()
        self.n_params = n_params
        self.n_cont = n_cont
        self.n_valve = n_valve
        self.T = T

        # ── Shared trunk ─────────────────────────────────────────────────
        trunk_layers: list[nn.Module] = [
            nn.Linear(n_params, hidden),
            nn.SiLU(),
        ]
        for _ in range(n_layers - 1):
            trunk_layers.append(nn.Linear(hidden, hidden))
            trunk_layers.append(nn.SiLU())
        self.trunk = nn.Sequential(*trunk_layers)

        # ── Output heads ─────────────────────────────────────────────────
        self.cont_head = nn.Linear(hidden, n_cont * T)
        self.valve_head = nn.Linear(hidden, n_valve * T)

    def forward(
        self,
        params: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(params)                           # (B, hidden)

        cont = self.cont_head(h).view(-1, self.n_cont, self.T)
        valve = self.valve_head(h).view(-1, self.n_valve, self.T)

        return cont, valve


class CVLoss(nn.Module):
    """
    MSE on continuous waveforms + transition-weighted BCE on valve logits.

    Timesteps within `transition_radius` of a 0→1 or 1→0 edge in the target
    are upweighted by `transition_weight` so the model is penalised more for
    missing sharp valve transitions.

    `forward` returns (total, cont_loss, valve_loss) so each component can be
    logged separately.
    """

    def __init__(
        self,
        valve_weight: float = 1.0,
        transition_weight: float = 5.0,
        transition_radius: int = 2,
    ):
        super().__init__()
        self.mse = nn.MSELoss()
        self.valve_weight = valve_weight
        self.transition_weight = transition_weight
        self.transition_radius = transition_radius

    def _transition_weights(self, target_valve: torch.Tensor) -> torch.Tensor:
        # target_valve: (B, n_valve, T) — values in {0, 1}
        # detect edges: where consecutive timesteps differ
        edges = (target_valve[..., 1:] != target_valve[..., :-1]).float()  # (B, n_valve, T-1)
        # pad back to T and dilate by transition_radius using max-pooling
        edges = torch.nn.functional.pad(edges, (0, 1))                     # (B, n_valve, T)
        edges = edges.unsqueeze(1)                                          # (B, 1, n_valve, T)
        kernel = 2 * self.transition_radius + 1
        edges = torch.nn.functional.max_pool2d(
            edges, kernel_size=(1, kernel), stride=1, padding=(0, self.transition_radius)
        ).squeeze(1)                                                        # (B, n_valve, T)
        return 1.0 + (self.transition_weight - 1.0) * edges

    def forward(
        self,
        pred_cont: torch.Tensor,
        pred_valve: torch.Tensor,
        target_cont: torch.Tensor,
        target_valve: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_cont = self.mse(pred_cont, target_cont)

        weights = self._transition_weights(target_valve)
        bce_elementwise = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_valve, target_valve, reduction="none"
        )
        loss_valve = (weights * bce_elementwise).mean()

        total = loss_cont + self.valve_weight * loss_valve
        return total, loss_cont, loss_valve