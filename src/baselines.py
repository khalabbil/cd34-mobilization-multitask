"""
src/baselines.py
═══════════════════════════════════════════════════════════════════════
Baselines for head-to-head comparison with ARSv4.
  · XGBoost (Model A & B) — manuscript best performer
  · Ridge — linear reference
  · Manuscript MLP PyTorch replica — exact reproduction of Sec 2.5 of
    the 15 Nisan 2026 manuscript (R² 0.05–0.16 negative result).
═══════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor


# ── XGBoost ───────────────────────────────────────────────────────
def make_xgboost(cfg_b: dict, seed: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=cfg_b["n_estimators"],
        max_depth=cfg_b["max_depth"],
        learning_rate=cfg_b["learning_rate"],
        subsample=cfg_b["subsample"],
        colsample_bytree=cfg_b["colsample_bytree"],
        objective="reg:squarederror",
        random_state=seed,
        verbosity=0,
        tree_method="hist",
        n_jobs=1,
    )


# ── Ridge ─────────────────────────────────────────────────────────
def make_ridge(cfg_b: dict, seed: int) -> Ridge:
    return Ridge(alpha=cfg_b["alpha"], random_state=seed)


# ── Manuscript MLP replica ────────────────────────────────────────
class ManuscriptMLP(nn.Module):
    """
    Faithful PyTorch port of manuscript §2.5:
      - 2 hidden layers (64, 32)
      - ReLU activation, BatchNorm, 30% dropout
      - L2 regularization (handled by optimizer weight_decay)
      - Adam optimizer (NOT AdamW)
      - Early stopping on val loss (handled by trainer)
    """
    def __init__(self, n_features: int, dropout: float = 0.30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_manuscript_mlp(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    cfg_b: dict, seed: int, device: torch.device,
) -> Tuple[np.ndarray, ManuscriptMLP]:
    """Trains the replica MLP with Adam + L2 + early stopping; returns val predictions."""
    torch.manual_seed(seed); np.random.seed(seed)
    
    model = ManuscriptMLP(n_features=X_train.shape[1], dropout=cfg_b["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg_b["lr"], weight_decay=cfg_b["l2"])
    loss_fn = nn.MSELoss()
    
    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.float32, device=device)
    Xva = torch.tensor(X_val,   dtype=torch.float32, device=device)
    yva = torch.tensor(y_val,   dtype=torch.float32, device=device)
    
    best_val = float("inf"); best_state = None; patience_counter = 0
    batch_size = 32
    
    for epoch in range(200):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch_size):
            idx = perm[i:i+batch_size]
            if len(idx) < 2:                              # BN needs ≥2
                continue
            opt.zero_grad()
            pred = model(Xtr[idx])
            loss = loss_fn(pred, ytr[idx])
            loss.backward()
            opt.step()
        
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva), yva).item()
        
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > cfg_b["early_stop_patience"]:
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = model(Xva).cpu().numpy()
    return val_pred, model
