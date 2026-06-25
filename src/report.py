"""
src/report.py
═══════════════════════════════════════════════════════════════════════
Consolidate OOF predictions into manuscript-comparable tables and a
markdown summary report.
═══════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _r2(y, yp):
    from sklearn.metrics import r2_score
    return float(r2_score(y, yp))

def _mae(y, yp):
    from sklearn.metrics import mean_absolute_error
    return float(mean_absolute_error(y, yp))

def _rmse(y, yp):
    from sklearn.metrics import mean_squared_error
    return float(np.sqrt(mean_squared_error(y, yp)))


def generate_report(cfg: dict) -> None:
    out_dir = Path(cfg["run"]["output_dir"])
    
    rows = []
    for fw in ["A", "B"]:
        npz_path = out_dir / f"oof_predictions_{fw}.npz"
        if not npz_path.exists():
            print(f"[report] skipping {fw}: {npz_path} not found")
            continue
        d = np.load(npz_path)
        y_yield = d["y_yield"]; y_eff = d["y_eff"]
        ars_y   = d["ars_yield_q"][:, 1]                  # P50
        ars_e   = d["ars_eff_q"][:, 1]
        ars_yq_lo, ars_yq_hi = d["ars_yield_q"][:, 0], d["ars_yield_q"][:, 2]
        ars_eq_lo, ars_eq_hi = d["ars_eff_q"][:, 0],   d["ars_eff_q"][:, 2]
        
        # Manuscript-comparable yield table
        for model_name, pred in [
            ("Ridge",           d["ridge_yield"]),
            ("Manuscript MLP",  d["msmlp_yield"]),
            ("XGBoost",         d["xgb_yield"]),
            ("ARS v4 yield (P50)", ars_y),
        ]:
            rows.append({
                "framework": fw, "target": "yield (×10⁶/kg)",
                "model": model_name,
                "MAE":  _mae(y_yield, pred),
                "RMSE": _rmse(y_yield, pred),
                "R²":   _r2(y_yield, pred),
            })
        # Efficiency table (XGB + ARS only — novel target)
        for model_name, pred in [
            ("XGBoost",         d["xgb_eff"]),
            ("ARS v4 eff (P50)", ars_e),
        ]:
            rows.append({
                "framework": fw, "target": "efficiency (×10⁶/mL)",
                "model": model_name,
                "MAE":  _mae(y_eff, pred),
                "RMSE": _rmse(y_eff, pred),
                "R²":   _r2(y_eff, pred),
            })
        # Calibration row for ARS v4
        picp = float(((y_yield >= ars_yq_lo) & (y_yield <= ars_yq_hi)).mean())
        mpiw = float(np.mean(ars_yq_hi - ars_yq_lo))
        rows.append({
            "framework": fw, "target": "yield calibration",
            "model": "ARS v4 (PI80)",
            "MAE": np.nan, "RMSE": mpiw, "R²": picp,
        })
    
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "tables" / "comparison_table.csv", index=False)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    
    # markdown summary
    md = ["# ARS v4.0 — Results Summary\n"]
    md.append("## Comparison Table (manuscript-comparable + new efficiency endpoint)\n")
    md.append(table.to_markdown(index=False, floatfmt=".3f"))
    md.append("\n")
    
    # manuscript headline numbers reminder
    md.append("## Manuscript benchmark (15 Nisan 2026, for reference)\n")
    md.append("| Framework | XGBoost R² | XGBoost MAE | Manuscript MLP R² |")
    md.append("|---|---|---|---|")
    md.append("| Model A   | 0.53 | 1.71 | 0.05 |")
    md.append("| Model B   | 0.70 | 1.33 | 0.16 |")
    
    (out_dir / "report.md").write_text("\n".join(md))
    print(f"[report] wrote {out_dir / 'report.md'}")
    print("\n" + table.to_string(index=False))
