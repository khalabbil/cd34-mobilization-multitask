"""
run_pipeline.py
═══════════════════════════════════════════════════════════════════════
ARS_FULL v4 — CLI orchestrator.

Usage:
    python run_pipeline.py --config configs/v4_config.yaml --mode full
    python run_pipeline.py --config configs/v4_config.yaml --mode train --framework B
    python run_pipeline.py --config configs/v4_config.yaml --mode predict --donor examples/donor_47.json
    python run_pipeline.py --config configs/v4_config.yaml --mode sensitivity
    python run_pipeline.py --config configs/v4_config.yaml --mode report
═══════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import yaml


def load_cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def cmd_train(cfg: dict, framework: str | None) -> None:
    from src.crossval import run_cv
    frameworks = [framework] if framework else ["A", "B"]
    for fw in frameworks:
        print(f"\n{'═' * 70}\nTRAINING Model {fw}\n{'═' * 70}")
        run_cv(cfg, framework=fw)


def cmd_predict(cfg: dict, donor_path: str) -> None:
    import joblib
    import torch
    from src.crossval     import pick_device
    from src.model        import ARSv4
    from src.inference    import predict_donor_full, format_clinical_report

    out_dir = Path(cfg["run"]["output_dir"])
    device = pick_device(cfg)
    
    fold_models = joblib.load(out_dir / "models" / "ensemble_B.joblib")
    # flatten seed_states into individual models for ensemble averaging
    ensemble = []
    for fm in fold_models:
        for st in fm["seed_states"]:
            ensemble.append({
                "model_state": st, "imputer": fm["imputer"], "scaler": fm["scaler"],
                "n_features": fm["n_features"], "model_cfg": fm["model_cfg"],
            })
    print(f"[predict] loaded ensemble of {len(ensemble)} models")
    
    donor = json.loads(Path(donor_path).read_text())
    donor.setdefault("framework", "B")
    result = predict_donor_full(donor, ensemble, cfg, device)
    print(format_clinical_report(result, donor_id=donor.get("donor_id", "Donor")))
    
    out_json = out_dir / f"prediction_{donor.get('donor_id', 'donor')}.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"\n[predict] full result → {out_json}")


def cmd_sensitivity(cfg: dict) -> None:
    print("\n═══ SENSITIVITY ANALYSIS (Q3) ═══")
    # Train a parallel model targeting y_abs (absolute total CD34 ×10⁶)
    # Then compare permutation importance with the y_yield head.
    import copy
    import numpy as np
    import torch
    from src.data         import load_and_prepare, get_features
    from src.preprocessing import fit_preprocessor, apply_preprocessor
    from src.crossval     import pick_device, run_cv
    from src.sensitivity  import compute_feature_importance_via_perturbation, \
                                  compare_rankings, save_sensitivity_report
    
    # 1) Train abs model (lightweight: just XGBoost on y_abs for importance)
    from xgboost import XGBRegressor
    
    df = load_and_prepare(cfg)
    X, feat_names = get_features(df, "B", cfg)
    
    # Split once for sensitivity (full CV optional)
    from sklearn.model_selection import train_test_split
    tr_idx, va_idx = train_test_split(np.arange(len(df)), test_size=0.2,
                                       random_state=cfg["run"]["seed"])
    imp, sc, Xtr = fit_preprocessor(X[tr_idx], cfg)
    Xva = apply_preprocessor(X[va_idx], imp, sc)
    
    # Yield model
    model_y = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    model_y.fit(Xtr, df["y_yield_log"].iloc[tr_idx].to_numpy())
    
    # Abs model
    model_a = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    model_a.fit(Xtr, df["y_abs_log"].iloc[tr_idx].to_numpy())
    
    imp_yield = compute_feature_importance_via_perturbation(
        lambda X: model_y.predict(X), Xva, feat_names)
    imp_abs   = compute_feature_importance_via_perturbation(
        lambda X: model_a.predict(X), Xva, feat_names)
    
    merged = imp_yield.merge(imp_abs, on="feature", suffixes=("_yield", "_abs"))
    comp   = compare_rankings(imp_yield, imp_abs)
    save_sensitivity_report(comp, merged,
                             Path(cfg["run"]["output_dir"]) / "sensitivity")


def cmd_report(cfg: dict) -> None:
    from src.report import generate_report
    generate_report(cfg)


def cmd_baselines_only(cfg: dict) -> None:
    """Train only baselines (for quick smoke tests)."""
    print("[baselines] not yet a standalone path — use --mode train (includes baselines)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", required=True,
                    choices=["full", "train", "predict", "sensitivity", "report", "smoke"])
    ap.add_argument("--framework", choices=["A", "B"], default=None)
    ap.add_argument("--donor", default=None, help="Path to JSON donor profile (for --mode predict)")
    ap.add_argument("--smoke", action="store_true",
                    help="Override config for fast smoke run: 1 repeat × 5 fold × 1 seed, "
                         "max_epochs=50, SWA start=30 (~3 min vs ~45 min full).")
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    if args.smoke:
        cfg["split"]["n_repeats"]               = 1
        cfg["ensemble"]["n_seeds"]              = 1
        cfg["ensemble"]["seeds"]                = [42]
        cfg["train"]["max_epochs"]              = 50
        cfg["train"]["early_stop"]["patience"]  = 10
        cfg["train"]["swa"]["start_epoch"]      = 30
        cfg["run"]["output_dir"]                = "outputs_smoke"
        print("[smoke] config overrides → 1 repeat × 5 fold × 1 seed, "
              "max_epochs=50, swa_start=30, outputs/ → outputs_smoke/")

    t0 = time.time()
    if args.mode == "smoke":
        # Just validate data loading + model build
        from src.data  import load_and_prepare, get_features
        from src.model import ARSv4
        df = load_and_prepare(cfg)
        X, feats = get_features(df, "B", cfg)
        m = ARSv4(n_features=X.shape[1])
        print(f"[smoke] data shape={df.shape}, X.shape={X.shape}, model params={m.n_parameters():,}")
    elif args.mode == "train":
        cmd_train(cfg, args.framework)
    elif args.mode == "predict":
        if not args.donor:
            sys.exit("--donor JSON path required for --mode predict")
        cmd_predict(cfg, args.donor)
    elif args.mode == "sensitivity":
        cmd_sensitivity(cfg)
    elif args.mode == "report":
        cmd_report(cfg)
    elif args.mode == "full":
        cmd_train(cfg, None)              # both A and B
        cmd_sensitivity(cfg)
        cmd_report(cfg)
    print(f"\n[runtime] total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
