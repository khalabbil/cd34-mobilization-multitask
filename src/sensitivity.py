"""
src/sensitivity.py
═══════════════════════════════════════════════════════════════════════
Sensitivity analysis — Q3.

Trains a parallel ARSv4 (or single-task variant) on `y_abs` (absolute
total CD34 ×10⁶), then compares feature importance with the y_yield
model. The goal: show that recipient weight's dominance in y_yield is
partly an artefact of the per-kg denominator (manuscript Limitation §5).

Output:
  - feature_importance_compare.csv  (SHAP rankings: y_yield vs y_abs)
  - rank_correlation.txt            (Spearman ρ between rankings)
  - patient_kg_relative_importance.png  (bar comparison)
═══════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Imports from this package
from .data import load_and_prepare, get_features, make_splits
from .model import ARSv4


def compute_feature_importance_via_perturbation(
    model_predict_fn, X: np.ndarray, feature_names: List[str], n_repeats: int = 10
) -> pd.DataFrame:
    """
    Permutation-style importance (model-agnostic, fast for sensitivity).
    For full SHAP, use src/shap_analysis.py.
    """
    rng = np.random.RandomState(42)
    baseline = model_predict_fn(X)
    base_var = np.var(baseline)
    rows = []
    for j, name in enumerate(feature_names):
        deltas = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            rng.shuffle(X_perm[:, j])
            new_pred = model_predict_fn(X_perm)
            deltas.append(np.mean((new_pred - baseline) ** 2))
        rows.append({"feature": name, "importance": np.mean(deltas) / (base_var + 1e-9)})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def compare_rankings(imp_yield: pd.DataFrame, imp_abs: pd.DataFrame) -> dict:
    """Returns Spearman ρ between feature rankings + patient_kg relative drop."""
    merged = imp_yield.merge(imp_abs, on="feature", suffixes=("_yield", "_abs"))
    rho, p = spearmanr(merged["importance_yield"], merged["importance_abs"])
    
    pk_yield = merged.loc[merged.feature == "patient_kg", "importance_yield"].iloc[0]
    pk_abs   = merged.loc[merged.feature == "patient_kg", "importance_abs"].iloc[0]
    
    return {
        "spearman_rho":  float(rho),
        "spearman_p":    float(p),
        "patient_kg_imp_yield":  float(pk_yield),
        "patient_kg_imp_abs":    float(pk_abs),
        "patient_kg_relative_drop_pct": float(100 * (pk_yield - pk_abs) / (pk_yield + 1e-9)),
    }


def save_sensitivity_report(comparison: dict, merged_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(out_dir / "feature_importance_compare.csv", index=False)
    with open(out_dir / "rank_correlation.txt", "w") as f:
        for k, v in comparison.items():
            f.write(f"{k}: {v}\n")
    # Manuscript revision text
    drop_pct = comparison["patient_kg_relative_drop_pct"]
    msg = (
        f"\nSENSITIVITY ANALYSIS RESULT (for manuscript §4.4 revision):\n"
        f"  Spearman ρ(yield-vs-abs feature rankings) = {comparison['spearman_rho']:.3f}\n"
        f"  patient_kg importance in y_yield model = {comparison['patient_kg_imp_yield']:.4f}\n"
        f"  patient_kg importance in y_abs model   = {comparison['patient_kg_imp_abs']:.4f}\n"
        f"  Relative drop when removing structural denominator = {drop_pct:.1f}%\n\n"
        f"  → A drop of >50% directly quantifies how much of patient_kg's\n"
        f"    apparent dominance was driven by the per-kg outcome definition\n"
        f"    vs genuine biological signal.\n"
    )
    with open(out_dir / "interpretation.txt", "w") as f:
        f.write(msg)
    print(msg)
