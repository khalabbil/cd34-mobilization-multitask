"""
src/crossval.py
═══════════════════════════════════════════════════════════════════════
The orchestrator that loops 10 repeats × 5 folds × 5 seeds = 250 models,
collects OOF predictions, computes ensemble averages, and persists the
final stack to outputs/models/.

Also runs all baselines on the SAME folds for direct comparison.
═══════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import copy
import json
import time
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch

from .data         import load_and_prepare, get_features, make_splits
from .preprocessing import fit_preprocessor, apply_preprocessor
from .train        import train_one_fold
from .baselines    import make_xgboost, make_ridge, train_manuscript_mlp


# ── metrics ──────────────────────────────────────────────────────
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def quantile_coverage(y_true: np.ndarray, q_low: np.ndarray, q_high: np.ndarray) -> dict:
    inside = (y_true >= q_low) & (y_true <= q_high)
    return {
        "picp":  float(inside.mean()),
        "mpiw":  float(np.mean(q_high - q_low)),
    }


def pick_device(cfg) -> torch.device:
    pref = cfg["run"]["device"]
    if pref == "auto":
        if torch.cuda.is_available():  return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


# ── main loop ───────────────────────────────────────────────────
def run_cv(cfg: dict, framework: str = "B") -> dict:
    out_dir = Path(cfg["run"]["output_dir"])
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    
    device = pick_device(cfg)
    print(f"[cv] device={device}  framework=Model {framework}")
    
    df = load_and_prepare(cfg)
    X, feat_names = get_features(df, framework, cfg)
    y_yield_log = df["y_yield_log"].to_numpy()
    y_eff_log   = df["y_eff_log"].to_numpy()
    y_yield     = df["y_yield"].to_numpy()
    y_eff       = df["y_eff"].to_numpy()
    n = len(df)
    
    n_quantiles = len(cfg["model"]["heads"]["yield"]["quantiles"])
    
    # OOF prediction stacks (averaged across seeds per fold)
    oof_yield_q = np.full((n, n_quantiles), np.nan)
    oof_eff_q   = np.full((n, n_quantiles), np.nan)
    oof_xgb_y   = np.full(n, np.nan)
    oof_xgb_e   = np.full(n, np.nan)
    oof_ridge_y = np.full(n, np.nan)
    oof_msmlp_y = np.full(n, np.nan)

    fold_metrics = []
    fold_models  = []                                  # persist for inference
    
    t0 = time.time()
    seeds = cfg["ensemble"]["seeds"]
    
    splits = list(make_splits(df, cfg))
    total_folds = len(splits)
    
    for split_idx, (rep, fold, tr_idx, va_idx) in enumerate(splits):
        Xtr_raw, Xva_raw = X[tr_idx], X[va_idx]
        imp, sc, Xtr = fit_preprocessor(Xtr_raw, cfg)
        Xva = apply_preprocessor(Xva_raw, imp, sc)

        # ── ARSv4 ensemble across seeds ──
        seed_yield, seed_eff = [], []
        seed_states = []
        for s in seeds:
            model, info = train_one_fold(
                Xtr, y_yield_log[tr_idx], y_eff_log[tr_idx],
                Xva, y_yield_log[va_idx], y_eff_log[va_idx],
                cfg, seed=s, device=device,
            )
            model.eval()
            with torch.no_grad():
                out = model(torch.tensor(Xva, dtype=torch.float32, device=device))
            seed_yield.append(np.expm1(out["yield_q"].cpu().numpy()))
            seed_eff.append(  np.expm1(out["eff_q"].cpu().numpy()))
            seed_states.append({k: v.cpu().clone() for k, v in model.state_dict().items()})

        oof_yield_q[va_idx] = np.mean(seed_yield, axis=0)
        oof_eff_q[va_idx]   = np.mean(seed_eff,   axis=0)

        # ── XGBoost ──
        xgb_y = make_xgboost(cfg["baselines"]["xgboost_A"], seed=42 + rep)
        xgb_y.fit(Xtr, y_yield_log[tr_idx])
        oof_xgb_y[va_idx] = np.expm1(xgb_y.predict(Xva))

        xgb_e = make_xgboost(cfg["baselines"]["xgboost_A"], seed=42 + rep)
        xgb_e.fit(Xtr, y_eff_log[tr_idx])
        oof_xgb_e[va_idx] = np.expm1(xgb_e.predict(Xva))

        # ── Ridge ──
        ridge = make_ridge(cfg["baselines"]["ridge"], seed=42 + rep)
        ridge.fit(Xtr, y_yield_log[tr_idx])
        oof_ridge_y[va_idx] = np.expm1(ridge.predict(Xva))

        # ── Manuscript MLP replica ──
        msmlp_pred, _ = train_manuscript_mlp(
            Xtr, y_yield_log[tr_idx], Xva, y_yield_log[va_idx],
            cfg["baselines"]["manuscript_mlp_replica"], seed=42 + rep, device=device,
        )
        oof_msmlp_y[va_idx] = np.expm1(msmlp_pred)

        # ── fold-level metrics (yield, P50) ──
        fm_y = regression_metrics(y_yield[va_idx], oof_yield_q[va_idx, 1])
        fm_e = regression_metrics(y_eff[va_idx],   oof_eff_q[va_idx,   1])
        fm_x = regression_metrics(y_yield[va_idx], oof_xgb_y[va_idx])
        fm_r = regression_metrics(y_yield[va_idx], oof_ridge_y[va_idx])
        fm_m = regression_metrics(y_yield[va_idx], oof_msmlp_y[va_idx])
        cov  = quantile_coverage(y_yield[va_idx],
                                  oof_yield_q[va_idx, 0], oof_yield_q[va_idx, 2])

        fold_metrics.append({
            "repeat": rep, "fold": fold,
            "ars_yield_mae": fm_y["mae"], "ars_yield_r2": fm_y["r2"],
            "ars_eff_mae":   fm_e["mae"], "ars_eff_r2":   fm_e["r2"],
            "xgb_yield_mae": fm_x["mae"], "xgb_yield_r2": fm_x["r2"],
            "ridge_yield_r2": fm_r["r2"],
            "msmlp_yield_r2": fm_m["r2"],
            "picp_80":       cov["picp"], "mpiw_80": cov["mpiw"],
        })
        
        # persist one representative model per fold (seed-averaged → keep all seeds for ensemble)
        fold_models.append({
            "repeat": rep, "fold": fold,
            "seed_states": seed_states,
            "imputer": imp, "scaler": sc,
            "n_features": X.shape[1],
            "model_cfg": {
                "n_features":  X.shape[1],
                "emb_dim":     cfg["preprocess"]["ft_embedding_dim"],
                "trunk_dim":   cfg["model"]["trunk_dim"],
                "n_blocks":    cfg["model"]["n_resblocks"],
                "expansion":   cfg["model"]["resblock_expansion"],
                "dropout":     cfg["model"]["dropout"],
                "head_hidden": tuple(cfg["model"]["heads"]["yield"]["hidden"]),
                "quantiles":   tuple(cfg["model"]["heads"]["yield"]["quantiles"]),
            },
        })
        
        elapsed = time.time() - t0
        avg_per_fold = elapsed / (split_idx + 1)
        eta = avg_per_fold * (total_folds - split_idx - 1)
        print(f"[cv] {split_idx+1:3d}/{total_folds}  rep={rep} fold={fold}  "
              f"ARS_R²={fm_y['r2']:+.3f}  XGB_R²={fm_x['r2']:+.3f}  "
              f"PICP={cov['picp']:.2f}  "
              f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    # ── overall OOF metrics (manuscript-comparable) ──
    overall = {
        "framework": framework,
        "n_folds": total_folds,
        "ars_yield_overall":  regression_metrics(y_yield, oof_yield_q[:, 1]),
        "ars_eff_overall":    regression_metrics(y_eff,   oof_eff_q[:,   1]),
        "xgb_yield_overall":  regression_metrics(y_yield, oof_xgb_y),
        "xgb_eff_overall":    regression_metrics(y_eff,   oof_xgb_e),
        "ridge_yield_overall":regression_metrics(y_yield, oof_ridge_y),
        "msmlp_yield_overall":regression_metrics(y_yield, oof_msmlp_y),
        "picp_80":            float(np.mean((y_yield >= oof_yield_q[:,0]) &
                                            (y_yield <= oof_yield_q[:,2]))),
        "mpiw_80":            float(np.mean(oof_yield_q[:,2] - oof_yield_q[:,0])),
    }
    print("\n[cv] FINAL OOF metrics:")
    print(json.dumps(overall, indent=2))

    # ── persist ──
    np.savez(out_dir / f"oof_predictions_{framework}.npz",
             y_yield=y_yield, y_eff=y_eff,
             ars_yield_q=oof_yield_q, ars_eff_q=oof_eff_q,
             xgb_yield=oof_xgb_y, xgb_eff=oof_xgb_e,
             ridge_yield=oof_ridge_y, msmlp_yield=oof_msmlp_y)
    pd.DataFrame(fold_metrics).to_csv(out_dir / f"fold_metrics_{framework}.csv", index=False)
    with open(out_dir / f"overall_metrics_{framework}.json", "w") as f:
        json.dump(overall, f, indent=2)
    joblib.dump(fold_models, out_dir / "models" / f"ensemble_{framework}.joblib", compress=3)
    
    return overall
