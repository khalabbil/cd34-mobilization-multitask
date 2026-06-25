"""
src/losses.py
═══════════════════════════════════════════════════════════════════════
Pinball (quantile) loss + multitask aggregator.
═══════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
from typing import Dict, Sequence, Tuple

import torch


def pinball_loss(
    pred_quantiles: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor,
) -> torch.Tensor:
    """
    pred_quantiles : (B, n_q)
    target         : (B,)
    quantiles      : (n_q,)
    Returns mean pinball loss over all quantiles.
    """
    target = target.unsqueeze(-1)                       # (B, 1)
    diff = target - pred_quantiles                       # (B, n_q)
    return torch.maximum(quantiles * diff, (quantiles - 1.0) * diff).mean()


def multitask_loss(
    out: Dict[str, torch.Tensor],
    y_yield_log: torch.Tensor,
    y_eff_log:   torch.Tensor,
    quantiles:   torch.Tensor,
    w_yield:     float = 0.5,
    w_eff:       float = 0.5,
    w_mse_p50:   float = 0.05,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Composite loss for the two quantile heads + a small MSE anchor on P50.
    """
    pin_y = pinball_loss(out["yield_q"], y_yield_log, quantiles)
    pin_e = pinball_loss(out["eff_q"],   y_eff_log,   quantiles)

    p50_idx = int(torch.argmin(torch.abs(quantiles - 0.5)).item())
    mse = (out["yield_q"][:, p50_idx] - y_yield_log).pow(2).mean() \
        + (out["eff_q"][:,   p50_idx] - y_eff_log  ).pow(2).mean()

    total = w_yield * pin_y + w_eff * pin_e + w_mse_p50 * mse
    return total, {
        "loss_total":  float(total.detach()),
        "loss_pin_y":  float(pin_y.detach()),
        "loss_pin_e":  float(pin_e.detach()),
        "loss_mse":    float(mse.detach()),
    }
