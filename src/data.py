"""
src/data.py
═══════════════════════════════════════════════════════════════════════
Data loading, target derivation, vaccine-string parsing, splitting.
The vaccine column is the only nontrivial parse — values include:
   "0", "1", "2", "1-2", "1,2", "1-2-3", "1 2", missing, etc.
═══════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold


# ── vaccine parser ────────────────────────────────────────────────
_VAX_SPLIT = re.compile(r"[\s,\-/_;]+")

def parse_vaccine_string(s) -> Dict[str, int]:
    """
    "1"        → {biontech: 1, sinovac: 0, turkovac: 0}
    "1-2"      → {biontech: 1, sinovac: 1, turkovac: 0}
    "1,2"      → {biontech: 1, sinovac: 1, turkovac: 0}
    "1-2-3"    → {biontech: 1, sinovac: 1, turkovac: 1}
    "0"/NaN/"" → all zeros
    Robust to numeric and string inputs.
    """
    if pd.isna(s):
        return {"vacc_biontech": 0, "vacc_sinovac": 0, "vacc_turkovac": 0}
    s = str(s).strip()
    if s in {"", "0", "nan", "None"}:
        return {"vacc_biontech": 0, "vacc_sinovac": 0, "vacc_turkovac": 0}
    tokens = {t for t in _VAX_SPLIT.split(s) if t}
    return {
        "vacc_biontech": int("1" in tokens),
        "vacc_sinovac":  int("2" in tokens),
        "vacc_turkovac": int("3" in tokens),
    }


def derive_age_years(dob: pd.Series, ref_date: str) -> pd.Series:
    ref = pd.to_datetime(ref_date)
    return ((ref - pd.to_datetime(dob)).dt.days / 365.25).astype(float)


# ── main loader ───────────────────────────────────────────────────
def load_and_prepare(cfg: dict) -> pd.DataFrame:
    """Returns a single DataFrame with all features + 3 targets."""
    df = pd.read_excel(cfg["data"]["path"])

    # 1) Derive age FIRST (before dropping Date_of_birth)
    if "Date_of_birth" in df.columns:
        df["age_years"] = derive_age_years(
            df["Date_of_birth"], cfg["data"]["derive_age_ref_date"]
        )

    # 2) Sex recode
    df["sex_male"] = (df["Sex_1_female_2_male"] == 2).astype(int)
    df = df.drop(columns=["Sex_1_female_2_male"])

    # 3) Now safe to drop the configured raw columns
    drop = [c for c in cfg["data"]["drop_cols"] if c in df.columns]
    df = df.drop(columns=drop)

    # Vaccine parse → 3 binaries
    vcol = cfg["data"]["vaccine_col_parse"]["source_col"]
    if vcol in df.columns:
        vax_df = pd.DataFrame(df[vcol].apply(parse_vaccine_string).tolist())
        df = pd.concat([df.drop(columns=[vcol]), vax_df], axis=1)

    # ── DERIVE 3 TARGETS ──
    yield_col = cfg["data"]["target_yield_col"]
    df["y_yield"] = df[yield_col].astype(float)

    # Efficiency: ×10⁶ CD34 captured per mL of product
    df["y_eff"] = (df[yield_col] * df["patient_kg"]) / df["product_volume_total"]

    # Sensitivity: absolute total CD34 (×10⁶) — Q3
    if cfg["data"]["sensitivity"]["enabled"]:
        df["y_abs"] = df[yield_col] * df["patient_kg"]

    # log1p transforms (training scale)
    if cfg["data"]["log_transform"]:
        df["y_yield_log"] = np.log1p(df["y_yield"])
        df["y_eff_log"]   = np.log1p(df["y_eff"])
        if "y_abs" in df.columns:
            df["y_abs_log"] = np.log1p(df["y_abs"])

    # Yield quintile for stratified split
    df["yield_quintile"] = pd.qcut(df["y_yield_log"], q=5, labels=False, duplicates="drop")

    return df


def get_features(df: pd.DataFrame, framework: str, cfg: dict) -> Tuple[np.ndarray, List[str]]:
    feats = cfg["features"][f"model_{framework}"]
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise KeyError(f"Missing features for Model {framework}: {missing}")
    return df[feats].to_numpy(dtype=float), feats


def make_splits(df: pd.DataFrame, cfg: dict):
    """Yields (repeat_idx, fold_idx, train_idx, val_idx)."""
    sp = cfg["split"]
    rskf = RepeatedStratifiedKFold(
        n_splits=sp["n_splits"],
        n_repeats=sp["n_repeats"],
        random_state=cfg["run"]["seed"],
    )
    y_strat = df[sp["stratify_by"]].to_numpy()
    X_dummy = np.zeros((len(df), 1))
    for i, (tr, va) in enumerate(rskf.split(X_dummy, y_strat)):
        repeat_idx = i // sp["n_splits"]
        fold_idx   = i %  sp["n_splits"]
        yield repeat_idx, fold_idx, tr, va


# ── smoke test ────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(Path("configs/v4_config.yaml").read_text())
    df = load_and_prepare(cfg)
    print(f"Shape: {df.shape}")
    print(f"Targets: y_yield median={df.y_yield.median():.3f}, "
          f"y_eff median={df.y_eff.median():.3f}")
    if "y_abs" in df.columns:
        print(f"         y_abs median={df.y_abs.median():.1f} (×10⁶ total)")
    X_A, _ = get_features(df, "A", cfg)
    X_B, _ = get_features(df, "B", cfg)
    print(f"Model A: X.shape={X_A.shape}")
    print(f"Model B: X.shape={X_B.shape}")
    n_folds = sum(1 for _ in make_splits(df, cfg))
    print(f"Total CV folds: {n_folds}  (should be 50)")
