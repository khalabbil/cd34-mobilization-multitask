"""
make_tct_figures.py — Publication-quality TCT figures into figures_TCT_final/.
All values come from locked OOF arrays / locked_metrics_pooled.json.
Style: Arial/Helvetica, colorblind-safe, tight bbox, legend-driven (no chart titles),
no internal project labels. Outputs vector PDF + 300 dpi PNG (+ SVG where useful).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures_TCT_final"
OUT.mkdir(exist_ok=True)

# ---- global style ----
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
# Colorblind-safe (Okabe-Ito)
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "purple": "#CC79A7", "grey": "#999999", "black": "#000000"}

L = json.loads((ROOT / "outputs" / "locked_metrics_pooled.json").read_text())
BA = json.loads((ROOT / "outputs" / "locked_bland_altman_quintile.json").read_text())
d = np.load(ROOT / "outputs" / "oof_predictions_B.npz")
y_yield, y_eff = d["y_yield"], d["y_eff"]
mtl_yq, mtl_eq = d["ars_yield_q"], d["ars_eff_q"]
qy = np.load(ROOT / "outputs" / "qxgb_oof_predictions_B_y_yield_log.npz")["qxgb_q"]
qe = np.load(ROOT / "outputs" / "qxgb_oof_predictions_B_y_eff_log.npz")["qxgb_q"]

def save(fig, stem):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  {stem}.pdf/.png/.svg")


# =====================================================================
# FIGURE 1 — architecture schematic (clean, minimal text)
# =====================================================================
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    def box(x, y, w, h, t, fc, ec, fs=8.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.10",
                                    lw=1.1, edgecolor=ec, facecolor=fc))
        ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal")
    def arr(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=11, lw=1.0, color=CB["grey"]))
    box(0.2, 1.4, 1.8, 1.2, "Donor\nfeatures", "#EAF3FB", CB["blue"], bold=True)
    box(2.4, 1.4, 1.8, 1.2, "FT\nembedding", "#EAF7F1", CB["green"])
    box(4.6, 1.2, 2.2, 1.6, "Shared\nresidual\ntrunk", "#EAF3FB", CB["blue"], bold=True)
    box(7.4, 2.35, 2.2, 1.0, "Yield\nquantile head", "#F7EEF4", CB["purple"])
    box(7.4, 0.65, 2.2, 1.0, "Efficiency\nquantile head", "#F7EEF4", CB["purple"])
    box(10.0, 2.35, 1.8, 1.0, "P10 / P50 / P90\n(yield)", "#FDF3E6", CB["orange"], fs=8)
    box(10.0, 0.65, 1.8, 1.0, "P10 / P50 / P90\n(efficiency)", "#FDF3E6", CB["orange"], fs=8)
    arr(2.0, 2.0, 2.4, 2.0); arr(4.2, 2.0, 4.6, 2.0)
    arr(6.8, 2.2, 7.4, 2.85); arr(6.8, 1.8, 7.4, 1.15)
    arr(9.6, 2.85, 10.0, 2.85); arr(9.6, 1.15, 10.0, 1.15)
    save(fig, "Figure_1_TCT_architecture")


# =====================================================================
# FIGURE 2 — Model B performance, 2 panels (yield, efficiency), 95% CI
# =====================================================================
def fig2():
    models = ["Multi-task\nmodel", "Quantile-\nXGBoost", "XGBoost"]
    colors = [CB["blue"], CB["orange"], CB["green"]]
    yld = [L["pooled_R2"]["mtl_yield"], L["pooled_R2"]["qxgb_yield"], L["pooled_R2"]["xgb_yield"]]
    eff = [L["pooled_R2"]["mtl_eff"], L["pooled_R2"]["qxgb_eff"], L["pooled_R2"]["xgb_eff"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, data, lab in zip(axes, [yld, eff], ["Yield", "Efficiency"]):
        xs = np.arange(len(models))
        pts = [v[0] for v in data]
        lo = [v[0]-v[1] for v in data]; hi = [v[2]-v[0] for v in data]
        ax.bar(xs, pts, color=colors, width=0.6, edgecolor="white", zorder=2)
        ax.errorbar(xs, pts, yerr=[lo, hi], fmt="none", ecolor=CB["black"],
                    elinewidth=1.0, capsize=3, zorder=3)
        for x, v, up in zip(xs, pts, hi):
            ax.text(x, v + up + 0.025, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(xs); ax.set_xticklabels(models, fontsize=7.5)
        ax.set_ylim(0, 0.85); ax.set_ylabel("Out-of-fold $R^2$" if lab == "Yield" else "")
        ax.text(0.5, 0.97, lab, transform=ax.transAxes, ha="center", va="top",
                fontsize=9, fontweight="bold")
        ax.grid(axis="y", alpha=0.25, zorder=0)
    save(fig, "Figure_2_TCT_performance")


# =====================================================================
# FIGURE 3 — calibration: multi-task vs quantile-XGBoost (Model B)
# Single clean panel per endpoint; the message (MTL nominal, qXGB undercovers).
# =====================================================================
def fig3():
    qlev = [0.10, 0.50, 0.90]
    def emp(yt, q):
        return [float((yt <= q[:, k]).mean()) for k in range(3)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, yt, mq, qq, lab in [(axes[0], y_yield, mtl_yq, qy, "Yield"),
                                (axes[1], y_eff, mtl_eq, qe, "Efficiency")]:
        ax.plot([0, 1], [0, 1], ls="--", lw=0.9, color=CB["grey"], label="Ideal")
        ax.plot(qlev, emp(yt, mq), "o-", color=CB["blue"], ms=6, lw=1.5, label="Multi-task model")
        ax.plot(qlev, emp(yt, qq), "s-", color=CB["orange"], ms=6, lw=1.5, label="Quantile-XGBoost")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Nominal quantile")
        if lab == "Yield":
            ax.set_ylabel("Empirical coverage"); ax.legend(frameon=False, loc="upper left")
        ax.text(0.5, 0.97, lab, transform=ax.transAxes, ha="center", va="top",
                fontsize=9, fontweight="bold")
        ax.grid(alpha=0.25)
    # annotate PICP-80
    axes[0].text(0.97, 0.06,
                 f"PICP-80: MTL {L['PICP80']['mtl_yield']:.2f} / qXGB {L['PICP80']['qxgb_yield']:.2f}",
                 transform=axes[0].transAxes, ha="right", fontsize=7.5)
    axes[1].text(0.97, 0.06,
                 f"PICP-80: MTL {L['PICP80']['mtl_eff']:.2f} / qXGB {L['PICP80']['qxgb_eff']:.2f}",
                 transform=axes[1].transAxes, ha="right", fontsize=7.5)
    save(fig, "Figure_3_TCT_calibration")


# =====================================================================
# FIGURE 4 (Supplement S1) — SHAP top-8, readable clinical names
# =====================================================================
def fig4_supp():
    imp = pd.read_csv(ROOT / "outputs" / "tables" / "shap_importance_B.csv")
    pretty = {
        "patient_kg": "Recipient weight",
        "pre_apheresis_cd34_cell_count_total_microliter": "Pre-apheresis CD34+ count",
        "sex_male": "Donor sex",
        "height": "Donor height",
        "gcsf_procedure_count": "G-CSF administration count",
        "pre_apheresis_wbc_cell_count": "Pre-apheresis WBC",
        "pre_apheresis_mononucleer_cell_count": "Pre-apheresis MNC",
        "weight": "Donor weight",
    }
    imp["disp"] = imp["feature"].map(pretty)
    sub = imp.dropna(subset=["disp"]).copy()
    sub = sub.sort_values("deepshap_ars_yield", ascending=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = np.arange(len(sub))
    ax.barh(y - 0.2, sub["deepshap_ars_yield"], height=0.38, color=CB["blue"], label="Multi-task yield head")
    ax.barh(y + 0.2, sub["treeshap_xgb_yield"], height=0.38, color=CB["orange"], label="XGBoost (yield)")
    ax.set_yticks(y); ax.set_yticklabels(sub["disp"], fontsize=8)
    ax.set_xlabel("Mean |SHAP| (relative attribution)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    save(fig, "Figure_S1_TCT_SHAP")


# =====================================================================
# FIGURE 5 — nomogram, single target dose 5e6/kg, clean heatmap
# =====================================================================
def fig5():
    kg = np.linspace(40, 100, 300)
    effv = np.linspace(0.6, 3.2, 300)
    KG, EFF = np.meshgrid(kg, effv)
    T = 5.0
    V = T * KG / EFF  # required product volume, mL
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    im = ax.pcolormesh(KG, EFF, V, shading="auto", cmap="viridis", vmin=80, vmax=520)
    cs = ax.contour(KG, EFF, V, levels=[100, 150, 200, 300, 400], colors="white",
                    linewidths=0.8, alpha=0.85)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%d mL")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Required apheresis product volume, mL")
    # worked example
    ax.scatter([65], [2.03], s=120, marker="*", color="white", edgecolors="black",
               linewidths=1.2, zorder=5)
    ax.annotate("Worked example\n65 kg, eff 2.03 → ~160 mL", xy=(65, 2.03), xytext=(70, 2.7),
                fontsize=7.5, color="white",
                arrowprops=dict(arrowstyle="->", color="white", lw=1.0))
    ax.set_xlabel("Recipient weight, kg")
    ax.set_ylabel("Predicted collection efficiency,\n$\\times10^6$ CD34+/mL product")
    save(fig, "Figure_5_TCT_nomogram")


# =====================================================================
# FIGURE 6 — Bland-Altman, final labels, transparency
# =====================================================================
def fig6():
    dfd = pd.read_json(ROOT / "outputs" / "bland_altman_summary.json", typ="series")
    import yaml, sys
    sys.path.insert(0, str(ROOT))
    cfg = yaml.safe_load((ROOT / "configs" / "v4_config.yaml").read_text())
    from src.data import load_and_prepare
    df = load_and_prepare(cfg)
    eff_p50 = mtl_eq[:, 1]
    pk = df["patient_kg"].to_numpy(float); av = df["product_volume_total"].to_numpy(float)
    vpred = y_yield * pk / eff_p50
    diff = vpred - av; mean = (vpred + av) / 2
    mb, sd = BA["overall"]["bias"], BA["overall"]["sd"]
    lo, hi = BA["overall"]["loa_lo"], BA["overall"]["loa_hi"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.scatter(mean, diff, s=16, alpha=0.30, color=CB["blue"], edgecolors="none")
    ax.axhline(mb, color=CB["red"], lw=1.3, label=f"Mean bias {mb:.1f} mL")
    ax.axhline(lo, color=CB["red"], lw=0.9, ls="--")
    ax.axhline(hi, color=CB["red"], lw=0.9, ls="--",
               label=f"95% LoA [{lo:.0f}, {hi:.0f}] mL")
    ax.axhline(0, color=CB["grey"], lw=0.6)
    ax.set_xlabel("Mean of predicted and actual product volume, mL")
    ax.set_ylabel("Predicted − actual product volume, mL")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(alpha=0.2)
    save(fig, "Figure_6_TCT_BlandAltman")


if __name__ == "__main__":
    print("Generating TCT figures ->", OUT)
    fig1(); fig2(); fig3(); fig4_supp(); fig5(); fig6()
    print("done.")
