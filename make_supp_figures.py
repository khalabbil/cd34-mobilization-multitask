"""
make_supp_figures.py — generate the two missing supplementary figures:
  Figure S2 — Model A interval reliability (pre-G-CSF setting)
  Figure S3 — multi-target nomogram (product volume across target doses 2-8 x10^6/kg)
Figure S1 (SHAP) already exists (make_tct_figures.py). Same house style.
All values from locked OOF arrays.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures_TCT_final"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "purple": "#CC79A7", "grey": "#999999", "black": "#000000"}

def save(fig, stem):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  {stem}.pdf/.png/.svg")

# ---- load Model A out-of-fold predictions ----
A = np.load(ROOT / "outputs" / "oof_predictions_A.npz")
yA_yield, yA_eff = A["y_yield"], A["y_eff"]
qA_y, qA_e = A["ars_yield_q"], A["ars_eff_q"]   # (n,3): P10,P50,P90


# =====================================================================
# FIGURE S2 — Model A interval reliability (pre-G-CSF)
# =====================================================================
def fig_s2():
    qlev = [0.10, 0.50, 0.90]
    def emp(yt, q):
        return [float((yt <= q[:, k]).mean()) for k in range(3)]
    def picp80(yt, q):
        return float(((yt >= q[:, 0]) & (yt <= q[:, 2])).mean())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    panels = [(axes[0], yA_yield, qA_y, "Yield"),
              (axes[1], yA_eff, qA_e, "Product concentration")]
    for ax, yt, q, lab in panels:
        ax.plot([0, 1], [0, 1], ls="--", lw=0.9, color=CB["grey"], label="Ideal")
        ax.plot(qlev, emp(yt, q), "o-", color=CB["blue"], ms=6, lw=1.5,
                label="Multi-task model (Model A)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Nominal quantile")
        if lab == "Yield":
            ax.set_ylabel("Empirical coverage")
            ax.legend(frameon=False, loc="upper left")
        ax.text(0.5, 0.97, lab, transform=ax.transAxes, ha="center", va="top",
                fontsize=9, fontweight="bold")
        ax.text(0.97, 0.06, f"PICP-80: {picp80(yt, q):.2f}",
                transform=ax.transAxes, ha="right", fontsize=7.5)
        ax.grid(alpha=0.25)
    save(fig, "Figure_S2_TCT_ModelA_reliability")


# =====================================================================
# FIGURE S3 — multi-target nomogram (doses 2,3,4,5,6,8 x10^6/kg)
# =====================================================================
def fig_s3():
    doses = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    kg = np.linspace(40, 100, 240)
    effv = np.linspace(0.6, 3.2, 240)
    KG, EFF = np.meshgrid(kg, effv)
    VMAX = 600.0
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 6.0), sharex=True, sharey=True)
    im = None
    for ax, T in zip(axes.ravel(), doses):
        V = np.clip(T * KG / EFF, 0, VMAX)
        im = ax.pcolormesh(KG, EFF, V, shading="auto", cmap="viridis", vmin=0, vmax=VMAX,
                           rasterized=True)
        cs = ax.contour(KG, EFF, T * KG / EFF, levels=[100, 150, 200, 300, 400, 500],
                        colors="white", linewidths=0.7, alpha=0.85)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%d")
        ax.text(0.5, 0.94, f"Target {int(T)} " + r"$\times10^6$ CD34+/kg",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8.5, fontweight="bold", color="white")
    for ax in axes[1, :]:
        ax.set_xlabel("Recipient weight, kg")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted product CD34+\nconcentration, " + r"$\times10^6$/mL")
    fig.subplots_adjust(right=0.88, wspace=0.08, hspace=0.12)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Required apheresis product volume, mL  (clipped at 600)")
    save(fig, "Figure_S3_TCT_multitarget_nomogram")


if __name__ == "__main__":
    print("Generating supplementary figures ->", OUT)
    fig_s2(); fig_s3()
    print("done.")
