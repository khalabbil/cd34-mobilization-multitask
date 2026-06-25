"""
src/shap_analysis.py
═══════════════════════════════════════════════════════════════════════
Three-paradigm SHAP concordance analysis (blueprint §9.4 / manuscript
Limitation #6 addressed):

    TreeSHAP       on XGBoost (yield) — reference importance ranking
    DeepExplainer  on ARSv4 yield head (P50)
    KernelExplainer on ARSv4 efficiency head (P50)

For tractability with n_features=16 and n≈316, we compute SHAP on a
single representative ensemble member rather than on all 250 — ranking
stability rather than exact ensemble-average SHAP is what we report.

Outputs (under outputs/<sub>/shap/):
  - shap_values_xgb_yield.npy
  - shap_values_ars_yield_p50.npy
  - shap_values_ars_eff_p50.npy
  - importance_table.csv  (mean |SHAP| per feature, per method)
  - concordance.json      (pairwise Spearman ρ on rankings)
  - shap_bar_<method>.png
═══════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch

from .data import load_and_prepare, get_features
from .preprocessing import fit_preprocessor, apply_preprocessor
from .model import ARSv4
from .baselines import make_xgboost


# ── helpers ─────────────────────────────────────────────────────────
def _mean_abs(shap_vals: np.ndarray) -> np.ndarray:
    """shap_vals: (N, n_features). Returns per-feature mean |SHAP|."""
    return np.abs(shap_vals).mean(axis=0)


def _rank(values: np.ndarray) -> np.ndarray:
    """Higher value → rank 1 (most important)."""
    order = np.argsort(-values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    rho, _ = spearmanr(a, b)
    return float(rho)


def _bar_plot(values: np.ndarray, feat_names: List[str],
              title: str, out_png: Path, dpi: int = 300, top_k: int = None) -> None:
    import matplotlib.pyplot as plt
    order = np.argsort(values)
    if top_k:
        order = order[-top_k:]
    fig, ax = plt.subplots(figsize=(6, max(3, 0.25 * len(order))), dpi=dpi)
    ax.barh(np.array(feat_names)[order], values[order], color="C0")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png); fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


# ── main entry point ────────────────────────────────────────────────
def run_shap_analysis(cfg: dict, framework: str = "B") -> Dict[str, dict]:
    """
    Loads OOF-trained ARSv4 ensemble (representative seed) and refits a
    single XGBoost on the full dataset for TreeSHAP. Runs DeepSHAP and
    KernelSHAP on the ARS yield/eff P50 outputs.
    """
    import shap

    out_dir   = Path(cfg["run"]["output_dir"])
    shap_dir  = out_dir / "shap"
    fig_dir   = out_dir / "figures"
    tbl_dir   = out_dir / "tables"
    for d in (shap_dir, fig_dir, tbl_dir):
        d.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare(cfg)
    X, feat_names = get_features(df, framework, cfg)
    y_yield_log = df["y_yield_log"].to_numpy()
    y_eff_log   = df["y_eff_log"].to_numpy()

    # Use whole data for SHAP background; the OOF-trained ensemble member's
    # imputer/scaler give an honest preprocessing.
    ensemble_path = out_dir / "models" / f"ensemble_{framework}.joblib"
    if not ensemble_path.exists():
        raise FileNotFoundError(f"Run training first: {ensemble_path} missing")
    fold_models = joblib.load(ensemble_path)

    # Representative member: first fold's first seed
    rep = fold_models[0]
    imp, scaler = rep["imputer"], rep["scaler"]
    Xp = apply_preprocessor(X, imp, scaler)

    bg_n = cfg["shap"]["background_samples"]
    rng  = np.random.default_rng(cfg["run"]["seed"])
    bg_idx = rng.choice(len(Xp), size=min(bg_n, len(Xp)), replace=False)
    Xp_bg = Xp[bg_idx]

    summary: Dict[str, dict] = {"framework": framework, "n_features": len(feat_names),
                                "n_total": len(Xp), "n_background": len(Xp_bg)}

    # ── (1) TreeSHAP on XGBoost (refit on full data) ──
    print("[shap] (1/3) TreeSHAP on XGBoost (yield)...", flush=True)
    xgb = make_xgboost(cfg["baselines"]["xgboost_A"], seed=cfg["run"]["seed"])
    xgb.fit(Xp, y_yield_log)
    tree_explainer = shap.TreeExplainer(xgb)
    sv_xgb = tree_explainer.shap_values(Xp)
    np.save(shap_dir / "shap_values_xgb_yield.npy", sv_xgb)
    imp_xgb = _mean_abs(sv_xgb)

    # ── (2) DeepExplainer on ARSv4 yield head P50 ──
    print("[shap] (2/3) DeepExplainer on ARSv4 yield head P50...", flush=True)
    device = torch.device("cpu")  # SHAP DeepExplainer is most stable on CPU
    model = ARSv4(**rep["model_cfg"]).to(device)
    model.load_state_dict(rep["seed_states"][0])
    model.eval()

    p50_idx = int(np.argmin(np.abs(np.array(rep["model_cfg"]["quantiles"]) - 0.5)))

    class YieldP50Wrapper(torch.nn.Module):
        def __init__(self, base, p50_idx):
            super().__init__(); self.base = base; self.p50_idx = p50_idx
        def forward(self, x):
            return self.base(x)["yield_q"][:, self.p50_idx:self.p50_idx + 1]

    class EffP50Wrapper(torch.nn.Module):
        def __init__(self, base, p50_idx):
            super().__init__(); self.base = base; self.p50_idx = p50_idx
        def forward(self, x):
            return self.base(x)["eff_q"][:, self.p50_idx:self.p50_idx + 1]

    yield_wrapper = YieldP50Wrapper(model, p50_idx).to(device).eval()
    eff_wrapper   = EffP50Wrapper(model,   p50_idx).to(device).eval()

    bg_tensor = torch.tensor(Xp_bg, dtype=torch.float32, device=device)
    X_tensor  = torch.tensor(Xp,    dtype=torch.float32, device=device)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        deep_expl = shap.DeepExplainer(yield_wrapper, bg_tensor)
        sv_ars_y = deep_expl.shap_values(X_tensor, check_additivity=False)
    if isinstance(sv_ars_y, list):
        sv_ars_y = sv_ars_y[0]
    sv_ars_y = np.asarray(sv_ars_y).reshape(len(Xp), -1)
    np.save(shap_dir / "shap_values_ars_yield_p50.npy", sv_ars_y)
    imp_ars_y = _mean_abs(sv_ars_y)

    # ── (3) KernelExplainer on ARSv4 efficiency head P50 ──
    print("[shap] (3/3) KernelExplainer on ARSv4 efficiency head P50...", flush=True)

    def eff_predict_numpy(x_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
            return eff_wrapper(x_t).cpu().numpy().ravel()

    kernel_bg = shap.sample(Xp_bg, min(50, len(Xp_bg)), random_state=cfg["run"]["seed"])
    kernel_expl = shap.KernelExplainer(eff_predict_numpy, kernel_bg)
    # KernelSHAP is O(n_features^2) — keep n_test modest
    n_test_for_kernel = min(100, len(Xp))
    test_idx = rng.choice(len(Xp), size=n_test_for_kernel, replace=False)
    sv_ars_e = kernel_expl.shap_values(Xp[test_idx], nsamples=128, silent=True)
    sv_ars_e = np.asarray(sv_ars_e)
    np.save(shap_dir / "shap_values_ars_eff_p50.npy", sv_ars_e)
    imp_ars_e = _mean_abs(sv_ars_e)

    # ── Importance table ──
    imp_df = pd.DataFrame({
        "feature":               feat_names,
        "treeshap_xgb_yield":    imp_xgb,
        "deepshap_ars_yield":    imp_ars_y,
        "kernelshap_ars_eff":    imp_ars_e,
        "rank_treeshap":         _rank(imp_xgb),
        "rank_deepshap":         _rank(imp_ars_y),
        "rank_kernelshap":       _rank(imp_ars_e),
    }).sort_values("treeshap_xgb_yield", ascending=False)
    imp_df.to_csv(tbl_dir / f"shap_importance_{framework}.csv", index=False)

    # ── Concordance ──
    concordance = {
        "spearman_treeshap_vs_deepshap":  _spearman(imp_xgb,   imp_ars_y),
        "spearman_treeshap_vs_kernelshap": _spearman(imp_xgb,  imp_ars_e),
        "spearman_deepshap_vs_kernelshap": _spearman(imp_ars_y, imp_ars_e),
    }
    summary["concordance"]    = concordance
    summary["top5_treeshap"]  = imp_df["feature"].head(5).tolist()
    summary["top5_deepshap"]  = imp_df.sort_values("deepshap_ars_yield", ascending=False)["feature"].head(5).tolist()
    summary["top5_kernelshap"] = imp_df.sort_values("kernelshap_ars_eff", ascending=False)["feature"].head(5).tolist()

    # ── Plots ──
    _bar_plot(imp_xgb,  feat_names, f"TreeSHAP · XGBoost yield (Model {framework})",
              fig_dir / f"shap_bar_treeshap_{framework}.png", cfg["report"]["dpi"])
    _bar_plot(imp_ars_y, feat_names, f"DeepSHAP · ARSv4 yield P50 (Model {framework})",
              fig_dir / f"shap_bar_deepshap_{framework}.png", cfg["report"]["dpi"])
    _bar_plot(imp_ars_e, feat_names, f"KernelSHAP · ARSv4 efficiency P50 (Model {framework})",
              fig_dir / f"shap_bar_kernelshap_{framework}.png", cfg["report"]["dpi"])

    with open(out_dir / f"shap_summary_{framework}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[shap] done. Concordance: {concordance}")
    return summary


if __name__ == "__main__":
    import sys, yaml
    cfg = yaml.safe_load(Path(sys.argv[1] if len(sys.argv) > 1
                              else "configs/v4_config.yaml").read_text())
    fw = sys.argv[2] if len(sys.argv) > 2 else "B"
    s = run_shap_analysis(cfg, framework=fw)
    print(json.dumps(s, indent=2))
