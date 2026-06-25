"""
src/inference.py
═══════════════════════════════════════════════════════════════════════
Donor-level inference + V_needed (volume to reach target dose) calculator.

Loads the saved ensemble from outputs/models/ and produces:
  - point + 80% PI predictions for yield (×10⁶/kg) and efficiency (×10⁶/mL)
  - volume table for a list of target doses
  - "will reach ≥ T/kg" verdict using P10 lower bound
═══════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .model import ARSv4
from .preprocessing import apply_preprocessor


def _back_transform_log1p(q: np.ndarray) -> np.ndarray:
    return np.expm1(q)


def predict_ensemble(
    X_raw: np.ndarray, ensemble: List[dict], device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    ensemble: list of dicts each with keys
        {model_state, imputer, scaler, n_features, model_cfg}
    Returns: ensemble-averaged P10/P50/P90 on original scale for both heads.
    """
    yields, effs = [], []
    for m in ensemble:
        X = apply_preprocessor(X_raw, m["imputer"], m["scaler"])
        model = ARSv4(**m["model_cfg"]).to(device)
        model.load_state_dict(m["model_state"])
        model.eval()
        with torch.no_grad():
            out = model(torch.tensor(X, dtype=torch.float32, device=device))
        yields.append(_back_transform_log1p(out["yield_q"].cpu().numpy()))
        effs.append(  _back_transform_log1p(out["eff_q"].cpu().numpy()))
    return {
        "yield_q": np.mean(yields, axis=0),               # (B, 3)
        "eff_q":   np.mean(effs,   axis=0),
        "yield_q_std": np.std(yields, axis=0),
        "eff_q_std":   np.std(effs,   axis=0),
    }


def compute_volume_table(
    eff_q: np.ndarray, patient_kg: float, target_doses: List[float],
    pi_low_idx: int = 0, pi_med_idx: int = 1, pi_high_idx: int = 2,
    clip_low: float = 50.0, clip_high: float = 800.0,
) -> List[dict]:
    """
    eff_q : (3,) array of [P10, P50, P90] for efficiency (×10⁶/mL)
    Returns volume rows for each target dose T (×10⁶/kg).
    """
    rows = []
    for T in target_doses:
        v_med = T * patient_kg / eff_q[pi_med_idx]
        # When efficiency is HIGH (eff P90 large) → volume LOW → optimistic
        v_lo  = T * patient_kg / eff_q[pi_high_idx]
        # When efficiency is LOW (eff P10 small) → volume HIGH → conservative
        v_hi  = T * patient_kg / eff_q[pi_low_idx]
        rows.append({
            "target_dose_per_kg": T,
            "V_median_mL":        round(float(np.clip(v_med, clip_low, clip_high)), 0),
            "V_PI80_low":         round(float(np.clip(v_lo,  clip_low, clip_high)), 0),
            "V_PI80_high":        round(float(np.clip(v_hi,  clip_low, clip_high)), 0),
        })
    return rows


def predict_donor_full(
    donor_dict: dict, ensemble: List[dict], cfg: dict, device: torch.device,
) -> dict:
    """
    Full clinical inference for a single donor (or batch of donors).
    donor_dict can be either a dict (single donor) or a list of dicts.
    """
    import pandas as pd
    if isinstance(donor_dict, dict):
        df = pd.DataFrame([donor_dict])
    else:
        df = pd.DataFrame(donor_dict)
    framework = donor_dict.get("framework", "B") if isinstance(donor_dict, dict) else "B"
    feats = cfg["features"][f"model_{framework}"]
    X_raw = df[feats].to_numpy(dtype=float)

    preds = predict_ensemble(X_raw, ensemble, device)
    
    results = []
    for i in range(len(df)):
        yq = preds["yield_q"][i]
        eq = preds["eff_q"][i]
        pkg = float(df["patient_kg"].iloc[i])
        vol_tbl = compute_volume_table(
            eq, pkg, cfg["inference"]["target_doses"],
            clip_low=cfg["inference"]["min_volume_mL"],
            clip_high=cfg["inference"]["max_volume_mL"],
        )
        results.append({
            "patient_kg": pkg,
            "yield_prediction": {
                "P10": float(yq[0]), "P50": float(yq[1]), "P90": float(yq[2]),
                "ensemble_std_P50": float(preds["yield_q_std"][i, 1]),
            },
            "efficiency_prediction": {
                "P10": float(eq[0]), "P50": float(eq[1]), "P90": float(eq[2]),
                "ensemble_std_P50": float(preds["eff_q_std"][i, 1]),
            },
            "volume_table": vol_tbl,
            "will_reach_5_per_kg": bool(yq[0] >= 5.0),    # P10 ≥ 5 → confident yes
            "will_reach_2_per_kg": bool(yq[0] >= 2.0),    # minimum acceptable
        })
    return results if len(results) > 1 else results[0]


def format_clinical_report(result: dict, donor_id: str = "Donor") -> str:
    """Human-readable text report (Turkish-friendly)."""
    lines = [f"═══ {donor_id} — CD34+ Mobilization Prediction ═══"]
    y = result["yield_prediction"]
    e = result["efficiency_prediction"]
    lines.append(f"  Patient (recipient) weight: {result['patient_kg']:.1f} kg")
    lines.append("")
    lines.append("  YIELD (×10⁶ CD34/kg recipient):")
    lines.append(f"    P50  = {y['P50']:.2f}   PI80 = [{y['P10']:.2f}, {y['P90']:.2f}]")
    lines.append(f"  EFFICIENCY (×10⁶ CD34/mL product):")
    lines.append(f"    P50  = {e['P50']:.2f}   PI80 = [{e['P10']:.2f}, {e['P90']:.2f}]")
    lines.append("")
    verdict_5 = "✓ YES (P10≥5)" if result["will_reach_5_per_kg"] else "✗ uncertain"
    verdict_2 = "✓ YES (P10≥2)" if result["will_reach_2_per_kg"] else "✗ POOR mobilizer risk"
    lines.append(f"  Will reach ≥5/kg: {verdict_5}")
    lines.append(f"  Will reach ≥2/kg: {verdict_2}")
    lines.append("")
    lines.append("  PRODUCT VOLUME NEEDED:")
    lines.append(f"  {'Target dose':>14} | {'V_median':>10} | {'PI80 (low–high)':>20}")
    lines.append(f"  {'×10⁶/kg':>14} | {'mL':>10} | {'mL':>20}")
    lines.append("  " + "-" * 50)
    for row in result["volume_table"]:
        T = row["target_dose_per_kg"]
        lines.append(f"  {T:>14.1f} | {row['V_median_mL']:>10.0f} | "
                     f"{row['V_PI80_low']:>9.0f} – {row['V_PI80_high']:<9.0f}")
    lines.append("═" * 50)
    return "\n".join(lines)
