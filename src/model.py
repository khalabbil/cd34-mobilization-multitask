"""
src/model.py
═══════════════════════════════════════════════════════════════════════
ARSv4 — Multi-task TabResNet with FT-style numerical embeddings.
Architecture rationale: see docs/v4_plan.md §5.

Output: dict with keys "yield_q" and "eff_q", each (B, n_quantiles).
Quantiles are monotonic by construction (cumsum trick).
═══════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
from typing import Dict, List, Sequence

import torch
import torch.nn as nn


class FTEmbedding(nn.Module):
    """Per-feature learnable linear projection (FT-Transformer's numerical block).

    Maps each scalar feature x_i ∈ ℝ to a vector w_i * x_i + b_i ∈ ℝ^d.
    Output shape: (B, n_features, d).
    """
    def __init__(self, n_features: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, dim) * 0.02)
        self.bias   = nn.Parameter(torch.zeros(n_features, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features) → (B, n_features, dim)
        return x.unsqueeze(-1) * self.weight + self.bias


class ResBlock(nn.Module):
    """Pre-LN residual block, expansion-then-projection (Transformer-FFN style)."""
    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.30):
        super().__init__()
        self.ln   = nn.LayerNorm(dim)
        self.lin1 = nn.Linear(dim, dim * expansion)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.lin2 = nn.Linear(dim * expansion, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.lin1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.lin2(h)
        return x + h


class QuantileHead(nn.Module):
    """Outputs monotone quantiles via cumsum(non-negative increments)."""
    def __init__(self, d_in: int, hidden: Sequence[int], n_quantiles: int, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = [nn.LayerNorm(d_in)]
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_quantiles))
        self.net = nn.Sequential(*layers)
        self.n_quantiles = n_quantiles

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)                                # (B, n_q)
        # First quantile = output[0]; subsequent = output[0] + cumsum(softplus(rest))
        first = out[:, :1]
        rest  = torch.nn.functional.softplus(out[:, 1:])  # ≥ 0 increments
        return torch.cat([first, first + torch.cumsum(rest, dim=1)], dim=1)


class ARSv4(nn.Module):
    """Multi-task tabular ResNet with two quantile heads."""
    def __init__(
        self,
        n_features: int,
        emb_dim: int = 8,
        trunk_dim: int = 64,
        n_blocks: int = 3,
        expansion: int = 2,
        dropout: float = 0.30,
        head_hidden: Sequence[int] = (32,),
        quantiles: Sequence[float] = (0.10, 0.50, 0.90),
    ):
        super().__init__()
        self.quantiles = torch.tensor(list(quantiles), dtype=torch.float32)
        self.emb  = FTEmbedding(n_features, emb_dim)
        self.proj = nn.Linear(n_features * emb_dim, trunk_dim)
        self.trunk = nn.Sequential(*[
            ResBlock(trunk_dim, expansion=expansion, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.head_yield = QuantileHead(trunk_dim, head_hidden, len(quantiles))
        self.head_eff   = QuantileHead(trunk_dim, head_hidden, len(quantiles))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.emb(x).flatten(1)
        h = self.proj(h)
        h = self.trunk(h)
        return {"yield_q": self.head_yield(h), "eff_q": self.head_eff(h)}

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── smoke test ────────────────────────────────────────────────────
if __name__ == "__main__":
    m = ARSv4(n_features=16)
    x = torch.randn(8, 16)
    out = m(x)
    print(f"Parameters: {m.n_parameters():,}")
    print(f"yield_q shape: {out['yield_q'].shape}")
    print(f"eff_q shape:   {out['eff_q'].shape}")
    # Monotonicity check
    q = out["yield_q"]
    assert (q[:, 1:] - q[:, :-1] >= 0).all(), "Quantile monotonicity violated"
    print("Quantile monotonicity: OK ✓")
