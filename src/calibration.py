"""
src/calibration.py
═══════════════════════════════════════════════════════════════════════
Quantile calibration analysis for ARSv4:
  - PICP-80 (Prediction Interval Coverage Probability, target 0.80)
  - MPIW    (Mean Prediction Interval Width)
  - Reliability diagram (empirical vs nominal quantile coverage)
  - Quantile crossing audit (should be 0 with cumsum head)
  - Binary endpoint metrics for clinical threshold (≥5 ×10⁶/kg)

Consumes outputs/oof_predictions_<framework>.npz produced by crossval.py.
═══════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ── core metrics ────────────────────────────────────────────────────
def picp_mpiw(y: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> Tuple[float, float]:
    inside = (y >= q_lo) & (y <= q_hi)
    return float(inside.mean()), float(np.mean(q_hi - q_lo))


def empirical_quantile_coverage(
    y: np.ndarray, q_pred: np.ndarray, quantiles: List[float]
) -> pd.DataFrame:
    """
    For each nominal quantile q_k, empirical fraction of y ≤ q_pred[:, k].
    A well-calibrated model produces empirical coverage ≈ nominal level.
    """
    rows = []
    for k, q in enumerate(quantiles):
        emp = float((y <= q_pred[:, k]).mean())
        rows.append({"nominal_quantile": q, "empirical_coverage": emp,
                     "deviation": emp - q})
    return pd.DataFrame(rows)


def quantile_crossing_audit(q_pred: np.ndarray) -> int:
    """Return number of rows where quantiles are not monotone non-decreasing."""
    diffs = q_pred[:, 1:] - q_pred[:, :-1]
    return int((diffs < 0).any(axis=1).sum())


# ── binary endpoint (manuscript: suboptimal mobilizer < 5/kg) ───────
def binary_endpoint_metrics(
    y_true_continuous: np.ndarray, y_pred_p50: np.ndarray,
    y_pred_p10: np.ndarray, threshold: float = 5.0,
) -> Dict[str, float]:
    """
    Two binary classifiers from the quantile output:
      - Point-prediction-based: P50 ≥ threshold → predicted reach
      - Confidence-based:       P10 ≥ threshold → high-confidence reach
    """
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, brier_score_loss,
        confusion_matrix
    )
    y_bin = (y_true_continuous >= threshold).astype(int)
    res: Dict[str, float] = {}

    if len(np.unique(y_bin)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"),
                "brier": float("nan"), "n_positives": int(y_bin.sum())}

    # Use P50 directly as a continuous score (higher → more likely above thr)
    res["auroc_p50"] = float(roc_auc_score(y_bin, y_pred_p50))
    res["auprc_p50"] = float(average_precision_score(y_bin, y_pred_p50))
    # Probabilistic interpretation needs sigmoid-ish calibration — skip Brier
    # for raw P50; report margin instead
    res["margin_p50_minus_thr_mean"] = float(np.mean(y_pred_p50 - threshold))

    # P10 ≥ threshold as a confident-yes flag
    confident_yes = (y_pred_p10 >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_bin, confident_yes, labels=[0, 1]).ravel()
    res["confident_yes_sensitivity"]  = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    res["confident_yes_specificity"]  = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    res["confident_yes_n_flagged"]    = int(confident_yes.sum())
    res["n_positives"]                = int(y_bin.sum())
    res["n_total"]                    = int(len(y_bin))
    return res


# ── plotting ────────────────────────────────────────────────────────
def reliability_diagram(
    y: np.ndarray, q_pred: np.ndarray, quantiles: List[float],
    title: str, out_png: Path, dpi: int = 300,
) -> None:
    import matplotlib.pyplot as plt
    cov = empirical_quantile_coverage(y, q_pred, quantiles)
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=dpi)
    ax.plot([0, 1], [0, 1], "--", color="grey", label="ideal")
    ax.plot(cov["nominal_quantile"], cov["empirical_coverage"],
            marker="o", color="C0", label="observed")
    for _, r in cov.iterrows():
        ax.annotate(f"{r['empirical_coverage']:.2f}",
                    (r["nominal_quantile"], r["empirical_coverage"]),
                    textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.set_xlabel("Nominal quantile")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


# ── orchestrator (entry point for run_pipeline.py --mode report) ────
def analyze_calibration(cfg: dict) -> Dict[str, dict]:
    out_dir = Path(cfg["run"]["output_dir"])
    fig_dir = out_dir / "figures"
    tbl_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    quantiles = cfg["model"]["heads"]["yield"]["quantiles"]
    summary = {}

    for fw in ["A", "B"]:
        npz = out_dir / f"oof_predictions_{fw}.npz"
        if not npz.exists():
            continue
        d = np.load(npz)
        y_yield = d["y_yield"]
        y_eff   = d["y_eff"]
        yq      = d["ars_yield_q"]
        eq      = d["ars_eff_q"]

        # PICP / MPIW (80% PI)
        picp_y, mpiw_y = picp_mpiw(y_yield, yq[:, 0], yq[:, 2])
        picp_e, mpiw_e = picp_mpiw(y_eff,   eq[:, 0], eq[:, 2])

        # Empirical coverage per nominal quantile
        cov_y = empirical_quantile_coverage(y_yield, yq, quantiles)
        cov_e = empirical_quantile_coverage(y_eff,   eq, quantiles)

        # Crossing audit
        crossings_y = quantile_crossing_audit(yq)
        crossings_e = quantile_crossing_audit(eq)

        # Reliability diagrams
        reliability_diagram(
            y_yield, yq, quantiles,
            title=f"Model {fw} · yield head reliability",
            out_png=fig_dir / f"reliability_yield_{fw}.png",
            dpi=cfg["report"]["dpi"],
        )
        reliability_diagram(
            y_eff, eq, quantiles,
            title=f"Model {fw} · efficiency head reliability",
            out_png=fig_dir / f"reliability_eff_{fw}.png",
            dpi=cfg["report"]["dpi"],
        )

        # Binary endpoint (yield ≥ 5/kg) — clinically meaningful
        thr = cfg["evaluation"]["binary_endpoint"]["threshold"]
        bin_y = binary_endpoint_metrics(y_yield, yq[:, 1], yq[:, 0], threshold=thr)

        # Persist
        cov_y.to_csv(tbl_dir / f"reliability_yield_{fw}.csv", index=False)
        cov_e.to_csv(tbl_dir / f"reliability_eff_{fw}.csv",   index=False)

        summary[fw] = {
            "picp_yield": picp_y, "mpiw_yield": mpiw_y,
            "picp_eff":   picp_e, "mpiw_eff":   mpiw_e,
            "crossings_yield": crossings_y,
            "crossings_eff":   crossings_e,
            "binary_endpoint_yield_ge_5_per_kg": bin_y,
        }

    with open(out_dir / "calibration_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    import sys, yaml
    cfg = yaml.safe_load(Path(sys.argv[1] if len(sys.argv) > 1
                              else "configs/v4_config.yaml").read_text())
    s = analyze_calibration(cfg)
    print(json.dumps(s, indent=2))
