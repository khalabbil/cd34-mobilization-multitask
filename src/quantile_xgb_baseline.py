"""
src/quantile_xgb_baseline.py
═══════════════════════════════════════════════════════════════════════
P1 CRITICAL FIX (Stage 4 REVISE, Devil's Advocate finding):

Single-task quantile-XGBoost efficiency and yield baselines, trained
with target-specific hyperparameter search, evaluated on the IDENTICAL
50-fold CV scheme used by the multi-task ARS v4 model.

This is the fair comparator for the multi-task model's calibrated PI80
claim. Without it, the +73% efficiency R² gain reported in §3.3 of the
v1 manuscript is presented against an un-tuned XGBoost using yield-model
hyperparameters — Devil's Advocate's CRITICAL issue.

Outputs (under outputs/):
  - qxgb_oof_predictions_<framework>.npz
  - qxgb_overall_metrics_<framework>.json
  - qxgb_fold_metrics_<framework>.csv

Reads: data2.xlsx + configs/v4_config.yaml
═══════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from .data import load_and_prepare, get_features, make_splits
from .preprocessing import fit_preprocessor, apply_preprocessor


# Hyperparameter grid for target-specific search (modest grid to keep runtime sane).
HP_GRID = [
    dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8),
]
QUANTILES = [0.10, 0.50, 0.90]


def _xgb_for_quantile(hp: dict, quantile: float, seed: int = 42) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=quantile,
        tree_method="hist",
        n_jobs=1,
        verbosity=0,
        random_state=seed,
        **hp,
    )


def _fit_and_predict_quantile_xgb(
    Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, hp: dict, seed: int = 42
) -> np.ndarray:
    """Returns (n_va, 3) array of [P10, P50, P90] predictions on Xva."""
    preds = np.empty((Xva.shape[0], len(QUANTILES)))
    for i, q in enumerate(QUANTILES):
        m = _xgb_for_quantile(hp, q, seed=seed)
        m.fit(Xtr, ytr)
        preds[:, i] = m.predict(Xva)
    return preds


def _enforce_monotonicity(q_pred: np.ndarray) -> np.ndarray:
    """Sort each row ascending so q_pred[:, 0] <= q_pred[:, 1] <= q_pred[:, 2]."""
    return np.sort(q_pred, axis=1)


def _pick_best_hp(Xtr: np.ndarray, ytr: np.ndarray, inner_seed: int) -> dict:
    """
    Pick HP via a single 80/20 inner split on the training fold,
    optimizing pinball loss across the three target quantiles.
    """
    n = len(Xtr)
    rng = np.random.RandomState(inner_seed)
    idx = rng.permutation(n)
    cut = int(0.8 * n)
    tr_in, va_in = idx[:cut], idx[cut:]
    best_hp, best_loss = HP_GRID[0], np.inf
    for hp in HP_GRID:
        q_pred = _fit_and_predict_quantile_xgb(Xtr[tr_in], ytr[tr_in], Xtr[va_in], hp, seed=inner_seed)
        q_pred = _enforce_monotonicity(q_pred)
        # Pinball loss across the three quantiles
        loss = 0.0
        y_in = ytr[va_in]
        for j, q in enumerate(QUANTILES):
            diff = y_in - q_pred[:, j]
            loss += np.maximum(q * diff, (q - 1.0) * diff).mean()
        loss /= len(QUANTILES)
        if loss < best_loss:
            best_loss = loss
            best_hp = hp
    return best_hp


def run_quantile_xgb_cv(cfg: dict, framework: str, target: str) -> dict:
    """target: 'y_yield_log' or 'y_eff_log'"""
    out_dir = Path(cfg["run"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_and_prepare(cfg)
    X, _ = get_features(df, framework, cfg)
    y_log = df[target].to_numpy()
    y_raw = df[target.replace("_log", "")].to_numpy()
    n = len(df)
    splits = list(make_splits(df, cfg))
    total_folds = len(splits)

    oof_q = np.full((n, len(QUANTILES)), np.nan)
    fold_metrics: List[dict] = []
    hp_chosen: List[dict] = []

    t0 = time.time()
    for split_idx, (rep, fold, tr_idx, va_idx) in enumerate(splits):
        imp, sc, Xtr = fit_preprocessor(X[tr_idx], cfg)
        Xva = apply_preprocessor(X[va_idx], imp, sc)
        ytr, yva = y_log[tr_idx], y_log[va_idx]
        yva_raw = y_raw[va_idx]
        hp = _pick_best_hp(Xtr, ytr, inner_seed=42 + rep)
        hp_chosen.append(hp)
        q_pred_log = _fit_and_predict_quantile_xgb(Xtr, ytr, Xva, hp, seed=42 + rep)
        q_pred_log = _enforce_monotonicity(q_pred_log)
        q_pred = np.expm1(q_pred_log)
        oof_q[va_idx] = q_pred

        # Per-fold metrics on the median (P50)
        p50 = q_pred[:, 1]
        fm = {
            "repeat": rep, "fold": fold,
            "mae":  float(mean_absolute_error(yva_raw, p50)),
            "rmse": float(np.sqrt(mean_squared_error(yva_raw, p50))),
            "r2":   float(r2_score(yva_raw, p50)),
            "picp_80": float(((yva_raw >= q_pred[:, 0]) & (yva_raw <= q_pred[:, 2])).mean()),
            "mpiw_80": float(np.mean(q_pred[:, 2] - q_pred[:, 0])),
            "hp_max_depth": hp["max_depth"],
            "hp_lr": hp["learning_rate"],
            "hp_n_est": hp["n_estimators"],
        }
        fold_metrics.append(fm)
        elapsed = time.time() - t0
        eta = (elapsed / (split_idx + 1)) * (total_folds - split_idx - 1)
        print(f"[qxgb {framework}/{target}] {split_idx+1:3d}/{total_folds}  R²={fm['r2']:+.3f}  PICP={fm['picp_80']:.2f}  "
              f"hp={hp['max_depth']}/{hp['n_estimators']}@{hp['learning_rate']}  ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    overall = {
        "framework": framework,
        "target":    target,
        "n_folds":   total_folds,
        "qxgb_p50_overall": {
            "mae":  float(mean_absolute_error(y_raw, oof_q[:, 1])),
            "rmse": float(np.sqrt(mean_squared_error(y_raw, oof_q[:, 1]))),
            "r2":   float(r2_score(y_raw, oof_q[:, 1])),
        },
        "picp_80": float(np.mean((y_raw >= oof_q[:, 0]) & (y_raw <= oof_q[:, 2]))),
        "mpiw_80": float(np.mean(oof_q[:, 2] - oof_q[:, 0])),
        "hp_chosen_modal": {
            "max_depth": int(np.median([hp["max_depth"] for hp in hp_chosen])),
            "learning_rate": float(np.median([hp["learning_rate"] for hp in hp_chosen])),
            "n_estimators": int(np.median([hp["n_estimators"] for hp in hp_chosen])),
        },
    }
    np.savez(out_dir / f"qxgb_oof_predictions_{framework}_{target}.npz",
             y_raw=y_raw, qxgb_q=oof_q)
    pd.DataFrame(fold_metrics).to_csv(
        out_dir / f"qxgb_fold_metrics_{framework}_{target}.csv", index=False)
    Path(out_dir / f"qxgb_overall_metrics_{framework}_{target}.json").write_text(
        json.dumps(overall, indent=2))
    print(f"[qxgb {framework}/{target}] DONE — R²={overall['qxgb_p50_overall']['r2']:.3f}, "
          f"PICP-80={overall['picp_80']:.3f}")
    return overall


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v4_config.yaml")
    ap.add_argument("--framework", default="B", choices=["A", "B"])
    ap.add_argument("--target", default="y_eff_log",
                    choices=["y_yield_log", "y_eff_log"])
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    res = run_quantile_xgb_cv(cfg, args.framework, args.target)
    print(json.dumps(res, indent=2))
