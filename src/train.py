"""
src/train.py
═══════════════════════════════════════════════════════════════════════
Single-fold trainer for ARSv4: mixup + SWA + cosine warm restarts +
multitask quantile loss + early stopping.

The CV orchestrator (src/crossval.py) calls train_one_fold() for each
of the 250 (10 repeats × 5 folds × 5 seeds) model instances.
═══════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from .model import ARSv4
from .losses import multitask_loss


def _mixup_batch(x, y_yield, y_eff, alpha):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    y_y = lam * y_yield + (1 - lam) * y_yield[idx]
    y_e = lam * y_eff   + (1 - lam) * y_eff[idx]
    return x_mix, y_y, y_e


def train_one_fold(
    X_train: np.ndarray, y_yield_log_tr: np.ndarray, y_eff_log_tr: np.ndarray,
    X_val:   np.ndarray, y_yield_log_va: np.ndarray, y_eff_log_va: np.ndarray,
    cfg: dict, seed: int, device: torch.device,
) -> Tuple[ARSv4, Dict[str, float]]:

    torch.manual_seed(seed); np.random.seed(seed)
    
    mcfg = cfg["model"]
    tcfg = cfg["train"]
    
    quantiles = torch.tensor(mcfg["heads"]["yield"]["quantiles"],
                             dtype=torch.float32, device=device)

    model = ARSv4(
        n_features=X_train.shape[1],
        emb_dim=cfg["preprocess"]["ft_embedding_dim"],
        trunk_dim=mcfg["trunk_dim"],
        n_blocks=mcfg["n_resblocks"],
        expansion=mcfg["resblock_expansion"],
        dropout=mcfg["dropout"],
        head_hidden=tuple(mcfg["heads"]["yield"]["hidden"]),
        quantiles=tuple(mcfg["heads"]["yield"]["quantiles"]),
    ).to(device)

    optim = AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = CosineAnnealingWarmRestarts(
        optim, T_0=tcfg["cosine_t0"], T_mult=tcfg["cosine_t_mult"],
        eta_min=tcfg["cosine_eta_min"],
    )

    swa_enabled = tcfg["swa"]["enabled"]
    swa_model = AveragedModel(model) if swa_enabled else None
    swa_scheduler = SWALR(optim, swa_lr=tcfg["swa"]["lr"]) if swa_enabled else None
    swa_start = tcfg["swa"]["start_epoch"]

    Xtr = torch.tensor(X_train,        dtype=torch.float32, device=device)
    yy  = torch.tensor(y_yield_log_tr, dtype=torch.float32, device=device)
    ye  = torch.tensor(y_eff_log_tr,   dtype=torch.float32, device=device)
    Xva = torch.tensor(X_val,          dtype=torch.float32, device=device)
    yyv = torch.tensor(y_yield_log_va, dtype=torch.float32, device=device)
    yev = torch.tensor(y_eff_log_va,   dtype=torch.float32, device=device)

    best_val = float("inf"); best_state = None; patience_ctr = 0
    bs = tcfg["batch_size"]
    
    history = []
    
    for epoch in range(tcfg["max_epochs"]):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i+bs]
            if len(idx) < 2: continue
            xb, yyb, yeb = Xtr[idx], yy[idx], ye[idx]
            
            if tcfg["mixup"]["enabled"] and epoch < swa_start:
                xb, yyb, yeb = _mixup_batch(xb, yyb, yeb, tcfg["mixup"]["alpha"])
            
            optim.zero_grad()
            out = model(xb)
            loss, _ = multitask_loss(
                out, yyb, yeb, quantiles,
                w_yield=tcfg["loss"]["pinball_yield_w"],
                w_eff  =tcfg["loss"]["pinball_eff_w"],
                w_mse_p50=tcfg["loss"]["mse_p50_w"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            optim.step()
        
        if swa_enabled and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            sched.step()
        
        # validation
        model.eval()
        with torch.no_grad():
            out_v = model(Xva)
            val_loss, parts = multitask_loss(
                out_v, yyv, yev, quantiles,
                w_yield=tcfg["loss"]["pinball_yield_w"],
                w_eff  =tcfg["loss"]["pinball_eff_w"],
                w_mse_p50=tcfg["loss"]["mse_p50_w"],
            )
            val_loss = float(val_loss)
        
        history.append({"epoch": epoch, **parts, "val_total": val_loss})

        if val_loss < best_val - tcfg["early_stop"]["min_delta"]:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr > tcfg["early_stop"]["patience"]:
                break

    # finalize SWA
    if swa_enabled and swa_model is not None and epoch >= swa_start:
        # BN update not strictly needed for LayerNorm models, but the
        # call is safe and gives a hook for future BN variants
        Xtr_loader = [(Xtr,)]                            # cheap "loader"
        try:
            update_bn(Xtr_loader, swa_model, device=device)
        except Exception:
            pass
        final_model = swa_model.module
    else:
        if best_state is not None:
            model.load_state_dict(best_state)
        final_model = model

    return final_model, {"best_val_loss": best_val, "n_epochs": len(history),
                         "history": history}
