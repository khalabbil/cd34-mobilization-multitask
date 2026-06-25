"""
cd34_decision_support_app.py
═══════════════════════════════════════════════════════════════════════
CD34+ Mobilization Decision Support — Multi-task TabResNet (Streamlit)

Four tabs:
  1. Single donor prediction  — 16-feature form → ensemble → V_needed nomogram
  2. Batch prediction         — upload donor table → batch inference → download
  3. Internal validation      — 316-donor OOF metrics (R², PICP, Bland-Altman)
  4. External validation      — upload a new labelled cohort → compute metrics
                                 and compare calibration drift vs the internal cohort

Run:
    cd ars_full
    .venv/bin/streamlit run app/cd34_decision_support_app.py
═══════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import sys
import json
import tempfile
from pathlib import Path
from copy import deepcopy

APP_DIR = Path(__file__).resolve().parent
ARS_ROOT = APP_DIR.parent
sys.path.insert(0, str(ARS_ROOT))

import numpy as np
import pandas as pd
import torch
import joblib
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from src.inference import predict_ensemble, compute_volume_table
from src.data import load_and_prepare, get_features

# ── config ──────────────────────────────────────────────────────────
st.set_page_config(page_title="CD34+ Mobilization Decision Support",
                    layout="wide", page_icon="🩸")

CFG = yaml.safe_load((ARS_ROOT / "configs" / "v4_config.yaml").read_text())
OUT = ARS_ROOT / "outputs"
DEVICE = torch.device("cpu")  # CPU inference: stable, fast enough for the app
TARGET_DOSES = CFG["inference"]["target_doses"]

FEATURES_B = CFG["features"]["model_B"]
FEATURES_A = CFG["features"]["model_A"]


# ── cached loaders ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading multi-task ensemble (250 models)…")
def load_ensemble(framework: str = "B"):
    path = OUT / "models" / f"ensemble_{framework}.joblib"
    if not path.exists():
        return None
    fold_models = joblib.load(path)
    ensemble = []
    for fm in fold_models:
        for state in fm["seed_states"]:
            ensemble.append({
                "model_state": state, "imputer": fm["imputer"], "scaler": fm["scaler"],
                "n_features": fm["n_features"], "model_cfg": fm["model_cfg"],
            })
    return ensemble


@st.cache_data(show_spinner=False)
def load_json(name: str):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_oof(framework: str = "B"):
    p = OUT / f"oof_predictions_{framework}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return {k: d[k] for k in d.files}


# ── metric helpers ──────────────────────────────────────────────────
def regression_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2":   float(r2_score(y_true, y_pred)),
    }


def picp_mpiw(y, q_lo, q_hi):
    inside = (y >= q_lo) & (y <= q_hi)
    return float(inside.mean()), float(np.mean(q_hi - q_lo))


def reliability_fig(y, q_pred, quantiles, title):
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    emp = [float((y <= q_pred[:, k]).mean()) for k in range(len(quantiles))]
    ax.plot([0, 1], [0, 1], "--", color="grey", label="ideal")
    ax.plot(quantiles, emp, "o-", color="#1A73E8", markersize=8, label="observed")
    ax.set_xlabel("Nominal quantile"); ax.set_ylabel("Empirical coverage")
    ax.set_title(title, fontsize=10); ax.legend(frameon=False, fontsize=8)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    return fig


def bland_altman_fig(v_pred, v_actual, title):
    diff = v_pred - v_actual
    mean = (v_pred + v_actual) / 2
    mb, sd = diff.mean(), diff.std(ddof=1)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(mean, diff, alpha=0.45, s=14, c="#1A73E8", edgecolor="none")
    ax.axhline(mb, color="#D32F2F", lw=1.4, label=f"bias = {mb:+.1f} mL")
    ax.axhline(mb - 1.96 * sd, color="#D32F2F", ls="--", lw=1.0)
    ax.axhline(mb + 1.96 * sd, color="#D32F2F", ls="--", lw=1.0)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("Mean of predicted and actual (mL)")
    ax.set_ylabel("Predicted − actual (mL)")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, mb, sd


# ── inference wrappers ──────────────────────────────────────────────
def run_inference(X_raw: np.ndarray, ensemble) -> dict:
    """X_raw: (N, n_features) raw feature matrix. Returns ensemble-averaged quantiles."""
    return predict_ensemble(X_raw, ensemble, DEVICE)


def derive_features_from_raw(uploaded_path: Path) -> pd.DataFrame:
    """Apply the same load_and_prepare pipeline used in training to a new raw Excel file."""
    cfg_local = deepcopy(CFG)
    cfg_local["data"]["path"] = str(uploaded_path)
    return load_and_prepare(cfg_local)


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
st.sidebar.title("CD34+ Mobilization\nDecision Support")
st.sidebar.markdown(
    """
**Model**: multi-task TabResNet
(yield + efficiency quantile heads)

**Cohort**: 316 healthy allogeneic donors
(Bursa Yüksek İhtisas EAH)

**Endpoints**
- Yield: ×10⁶ CD34+/kg recipient
- Efficiency: ×10⁶ CD34+/mL product

**Dose thresholds**
- ≥ 2 ×10⁶/kg — minimum acceptable
- ≥ 5 ×10⁶/kg — optimal target

⚠️ Research prototype. Not for
unsupervised clinical use.
"""
)
framework = st.sidebar.radio(
    "Feature framework",
    options=["B", "A"],
    format_func=lambda x: ("Model B — post-G-CSF (incl. pre-apheresis CD34)"
                            if x == "B" else "Model A — pre-G-CSF screening"),
)
ensemble = load_ensemble(framework)
if ensemble is None:
    st.error(f"Ensemble file outputs/models/ensemble_{framework}.joblib not found. "
             f"Run `python run_pipeline.py --mode train` first.")
    st.stop()
st.sidebar.success(f"Loaded {len(ensemble)} models (framework {framework})")

FEATURES = FEATURES_B if framework == "B" else FEATURES_A


# ════════════════════════════════════════════════════════════════════
# MAIN TABS
# ════════════════════════════════════════════════════════════════════
st.title("CD34+ Mobilization Decision Support")

tab1, tab2, tab3, tab4 = st.tabs([
    "🧍 Single donor", "📋 Batch prediction",
    "📊 Internal validation", "🌍 External validation",
])

# ── TAB 1 — SINGLE DONOR ────────────────────────────────────────────
with tab1:
    st.subheader("Single donor prediction")
    st.caption("Enter pre-apheresis donor features; the ensemble returns P10/P50/P90 "
               "for yield and efficiency plus a V_needed volume nomogram.")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age (years)", 18.0, 80.0, 35.0, 0.5)
        sex_male = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x else "Female")
        weight = st.number_input("Donor weight (kg)", 40.0, 150.0, 77.0, 0.5)
        height = st.number_input("Donor height (cm)", 140.0, 210.0, 174.0, 0.5)
        bmi = st.number_input("BMI (kg/m²)", 15.0, 50.0, 25.7, 0.1)
        patient_kg = st.number_input("Recipient weight (kg)", 5.0, 160.0, 70.0, 0.5)
    with c2:
        smoking = st.selectbox("Smoking status", [0, 1, 2],
                               format_func=lambda x: ["Never", "Ex-smoker", "Smoker"][x])
        pack_year = st.number_input("Smoking pack-years", 0.0, 100.0, 0.0, 1.0)
        gcsf_days = st.number_input("G-CSF day count", 1, 20, 9, 1)
        wbc = st.number_input("Pre-apheresis WBC (cells/µL)", 0.0, 200000.0, 52400.0, 100.0)
        mnc = st.number_input("Pre-apheresis MNC (cells/µL)", 0.0, 50000.0, 2800.0, 10.0)
    with c3:
        covid_vax = st.selectbox("COVID-19 vaccinated", [1, 0],
                                 format_func=lambda x: "Yes" if x else "No")
        covid_doses = st.number_input("COVID-19 vaccine doses", 0, 10, 2, 1)
        vacc_biontech = st.selectbox("BNT162b2 (BioNTech)", [1, 0],
                                     format_func=lambda x: "Yes" if x else "No")
        vacc_sinovac = st.selectbox("CoronaVac (Sinovac)", [0, 1],
                                    format_func=lambda x: "Yes" if x else "No")
        pre_cd34 = None
        if framework == "B":
            pre_cd34 = st.number_input("Pre-apheresis CD34+ (/µL)", 0.0, 1000.0, 81.8, 0.5)

    feature_values = {
        "age_years": age, "sex_male": sex_male, "weight": weight, "height": height,
        "bmi": bmi, "smoking_status_0_naiv_1_ex_smoker_2_smoker": smoking,
        "smoking_package_year": pack_year, "patient_kg": patient_kg,
        "pre_apheresis_wbc_cell_count": wbc,
        "pre_apheresis_mononucleer_cell_count": mnc,
        "pre_apheresis_cd34_cell_count_total_microliter": pre_cd34,
        "gcsf_procedure_count": gcsf_days, "covid19_vaccine_status": covid_vax,
        "covid19_vaccine_how_many_times": covid_doses,
        "vacc_biontech": vacc_biontech, "vacc_sinovac": vacc_sinovac,
    }

    if st.button("Predict", type="primary", key="single_predict"):
        X = np.array([[feature_values[f] for f in FEATURES]], dtype=float)
        preds = run_inference(X, ensemble)
        yq = preds["yield_q"][0]   # [P10, P50, P90]
        eq = preds["eff_q"][0]

        m1, m2, m3 = st.columns(3)
        m1.metric("Yield P50 (×10⁶/kg)", f"{yq[1]:.2f}", f"PI80 [{yq[0]:.2f}, {yq[2]:.2f}]")
        m2.metric("Efficiency P50 (×10⁶/mL)", f"{eq[1]:.2f}", f"PI80 [{eq[0]:.2f}, {eq[2]:.2f}]")
        reach5 = "✅ YES (P10 ≥ 5)" if yq[0] >= 5 else (
                 "⚠️ likely" if yq[1] >= 5 else "❌ at risk")
        m3.metric("Reaches ≥ 5 ×10⁶/kg?", reach5)

        if yq[0] < 2:
            st.error("⚠️ Poor-mobilizer risk: P10 yield below the 2 ×10⁶/kg minimum. "
                     "Consider alternative mobilization planning.")
        elif yq[0] < 5:
            st.warning("Suboptimal-mobilizer zone: P10 yield below the 5 ×10⁶/kg optimal "
                       "target. Plan apheresis volume with the conservative bound.")
        else:
            st.success("Confident adequate mobilizer: P10 yield at or above 5 ×10⁶/kg.")

        st.markdown("#### V_needed nomogram — product volume to reach a target dose")
        vol_rows = compute_volume_table(
            eq, patient_kg, TARGET_DOSES,
            clip_low=CFG["inference"]["min_volume_mL"],
            clip_high=CFG["inference"]["max_volume_mL"],
        )
        vol_df = pd.DataFrame(vol_rows)
        vol_df.columns = ["Target dose (×10⁶/kg)", "V median (mL)",
                          "V PI80 low (mL)", "V PI80 high (mL)"]
        st.dataframe(vol_df, width="stretch", hide_index=True)
        st.caption("V(T) = T × recipient_kg / efficiency. Low/high bounds use the "
                   "P90/P10 efficiency quantiles respectively.")


# ── TAB 2 — BATCH PREDICTION ────────────────────────────────────────
with tab2:
    st.subheader("Batch prediction")
    st.caption("Upload a donor table (raw Excel with the data2.xlsx column schema, or a "
               "pre-derived CSV with the 15/16 model features). Predictions are appended "
               "and the full table can be downloaded.")
    up = st.file_uploader("Donor table (.xlsx / .csv)", type=["xlsx", "csv"], key="batch_up")
    if up is not None:
        try:
            if up.name.endswith(".xlsx"):
                with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                    tmp.write(up.getvalue()); tmp_path = Path(tmp.name)
                df = derive_features_from_raw(tmp_path)
                st.info(f"Raw Excel detected — features derived via the training pipeline. "
                        f"{len(df)} donors.")
            else:
                df = pd.read_csv(up)
                st.info(f"CSV detected — {len(df)} rows. Expecting model features as columns.")
            missing = [f for f in FEATURES if f not in df.columns]
            if missing:
                st.error(f"Missing required feature columns: {missing}")
            else:
                X = df[FEATURES].to_numpy(dtype=float)
                preds = run_inference(X, ensemble)
                out_df = df.copy()
                out_df["yield_P10"] = preds["yield_q"][:, 0]
                out_df["yield_P50"] = preds["yield_q"][:, 1]
                out_df["yield_P90"] = preds["yield_q"][:, 2]
                out_df["eff_P10"]   = preds["eff_q"][:, 0]
                out_df["eff_P50"]   = preds["eff_q"][:, 1]
                out_df["eff_P90"]   = preds["eff_q"][:, 2]
                if "patient_kg" in out_df.columns:
                    out_df["V_5kg_median_mL"] = (5.0 * out_df["patient_kg"]
                                                  / out_df["eff_P50"]).round(0)
                out_df["reaches_5_per_kg"] = out_df["yield_P10"] >= 5.0
                st.success(f"Predicted {len(out_df)} donors.")
                st.dataframe(out_df.head(50), width="stretch")
                st.download_button(
                    "⬇ Download full predictions (CSV)",
                    out_df.to_csv(index=False).encode(),
                    file_name="cd34_batch_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Could not process file: {e}")


# ── TAB 3 — INTERNAL VALIDATION ─────────────────────────────────────
with tab3:
    st.subheader("Internal validation — 316-donor out-of-fold performance")
    st.caption("Metrics from the 10×5 repeated stratified cross-validation on the "
               "development cohort. These are fixed reference values.")

    gt = load_json(f"overall_metrics_{framework}.json")
    cal = load_json("calibration_summary.json")
    ba = load_json("bland_altman_summary.json")
    oof = load_oof(framework)

    if gt is None:
        st.warning(f"overall_metrics_{framework}.json not found.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ARS yield R²", f"{gt['ars_yield_overall']['r2']:.3f}")
        c2.metric("ARS efficiency R²", f"{gt['ars_eff_overall']['r2']:.3f}")
        c3.metric("PICP-80 (yield)", f"{gt['picp_80']:.3f}")
        c4.metric("MPIW-80 (yield)", f"{gt['mpiw_80']:.2f}")

        cmp_df = pd.DataFrame({
            "Model": ["Multi-task TabResNet", "XGBoost (default)", "Ridge", "Vanilla MLP"],
            "Yield R²": [gt["ars_yield_overall"]["r2"], gt["xgb_yield_overall"]["r2"],
                          gt["ridge_yield_overall"]["r2"], gt["msmlp_yield_overall"]["r2"]],
            "Yield MAE": [gt["ars_yield_overall"]["mae"], gt["xgb_yield_overall"]["mae"],
                           gt["ridge_yield_overall"]["mae"], gt["msmlp_yield_overall"]["mae"]],
        })
        st.markdown("##### Model comparison (out-of-fold)")
        st.dataframe(cmp_df, width="stretch", hide_index=True)

        if cal and framework in cal:
            be = cal[framework]["binary_endpoint_yield_ge_5_per_kg"]
            st.markdown("##### Binary endpoint — yield ≥ 5 ×10⁶/kg")
            b1, b2, b3 = st.columns(3)
            b1.metric("AUROC", f"{be['auroc_p50']:.3f}")
            b2.metric("AUPRC", f"{be['auprc_p50']:.3f}")
            b3.metric("Confident-yes specificity", f"{be['confident_yes_specificity']:.2f}")

        if oof is not None:
            quantiles = CFG["model"]["heads"]["yield"]["quantiles"]
            cc1, cc2 = st.columns(2)
            with cc1:
                st.pyplot(reliability_fig(oof["y_yield"], oof["ars_yield_q"], quantiles,
                                          f"Reliability — yield (Model {framework})"))
            with cc2:
                st.pyplot(reliability_fig(oof["y_eff"], oof["ars_eff_q"], quantiles,
                                          f"Reliability — efficiency (Model {framework})"))

        if ba:
            st.markdown("##### V_needed Bland-Altman (internal cohort)")
            st.write(f"Mean bias **{ba['mean_bias_mL']:.1f} mL**, "
                     f"95% LoA [{ba['loa_low_mL']:.1f}, {ba['loa_high_mL']:.1f}] mL "
                     f"(n = {ba['n']}).")


# ── TAB 4 — EXTERNAL VALIDATION ─────────────────────────────────────
with tab4:
    st.subheader("External validation — evaluate the multi-task model on a new labelled cohort")
    st.caption("Upload a new centre's data in the data2.xlsx column schema (must include "
               "the actual collected CD34 count and product volume so true outcomes can "
               "be derived). The app derives features, runs the ensemble, and reports "
               "performance + calibration drift against the internal cohort.")

    ext = st.file_uploader("External cohort (.xlsx, data2 schema)", type=["xlsx"],
                            key="ext_up")
    if ext is not None:
        try:
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                tmp.write(ext.getvalue()); ext_path = Path(tmp.name)
            df_ext = derive_features_from_raw(ext_path)
            n_ext = len(df_ext)
            st.success(f"External cohort loaded: {n_ext} donors. Features derived.")

            missing = [f for f in FEATURES if f not in df_ext.columns]
            if missing:
                st.error(f"Cannot run — missing derived features: {missing}")
            elif "y_yield" not in df_ext.columns or "y_eff" not in df_ext.columns:
                st.error("True outcomes (y_yield / y_eff) could not be derived. "
                         "The file must contain collected_cd34_cell_count, patient_kg, "
                         "and product_volume_total.")
            else:
                X = df_ext[FEATURES].to_numpy(dtype=float)
                preds = run_inference(X, ensemble)
                y_yield = df_ext["y_yield"].to_numpy()
                y_eff = df_ext["y_eff"].to_numpy()
                yq, eq = preds["yield_q"], preds["eff_q"]

                # Performance
                m_yield = regression_metrics(y_yield, yq[:, 1])
                m_eff   = regression_metrics(y_eff, eq[:, 1])
                picp_y, mpiw_y = picp_mpiw(y_yield, yq[:, 0], yq[:, 2])
                picp_e, mpiw_e = picp_mpiw(y_eff, eq[:, 0], eq[:, 2])

                # Internal reference
                gt_int = load_json(f"overall_metrics_{framework}.json")

                st.markdown("##### Performance — external vs internal")
                rows = [
                    ["Yield R²",        m_yield["R2"],  gt_int["ars_yield_overall"]["r2"]],
                    ["Yield MAE",       m_yield["MAE"], gt_int["ars_yield_overall"]["mae"]],
                    ["Efficiency R²",   m_eff["R2"],    gt_int["ars_eff_overall"]["r2"]],
                    ["Efficiency MAE",  m_eff["MAE"],   gt_int["ars_eff_overall"]["mae"]],
                    ["PICP-80 yield",   picp_y,         gt_int["picp_80"]],
                    ["PICP-80 eff",     picp_e,         None],
                ]
                cmp = pd.DataFrame(rows, columns=["Metric", "External", "Internal"])
                cmp["Δ (ext − int)"] = cmp.apply(
                    lambda r: (r["External"] - r["Internal"])
                    if r["Internal"] is not None else np.nan, axis=1)
                st.dataframe(cmp.round(3), width="stretch", hide_index=True)

                # Drift verdict
                r2_drop = gt_int["ars_yield_overall"]["r2"] - m_yield["R2"]
                if r2_drop > 0.15:
                    st.error(f"⚠️ Substantial performance drop on yield "
                             f"(ΔR² = −{r2_drop:.3f}). The model may not transfer to "
                             f"this centre without recalibration.")
                elif r2_drop > 0.05:
                    st.warning(f"Moderate performance drop on yield (ΔR² = −{r2_drop:.3f}). "
                               f"Interpret external predictions with caution.")
                else:
                    st.success(f"Yield performance largely retained "
                               f"(ΔR² = {-r2_drop:+.3f}). Calibration appears to transfer.")

                # Calibration drift
                if not (0.75 <= picp_y <= 0.85):
                    st.warning(f"PICP-80 (yield) = {picp_y:.3f} is outside the target band "
                               f"[0.75, 0.85] — prediction intervals are mis-calibrated on "
                               f"this cohort and should be re-conformalized.")

                cc1, cc2 = st.columns(2)
                quantiles = CFG["model"]["heads"]["yield"]["quantiles"]
                with cc1:
                    st.pyplot(reliability_fig(y_yield, yq, quantiles,
                                              "External reliability — yield"))
                with cc2:
                    # V_needed Bland-Altman on external cohort
                    if "patient_kg" in df_ext.columns and "product_volume_total" in df_ext.columns:
                        v_pred = y_yield * df_ext["patient_kg"].to_numpy() / eq[:, 1]
                        v_act = df_ext["product_volume_total"].to_numpy().astype(float)
                        fig_ba, mb, sd = bland_altman_fig(v_pred, v_act,
                                                          "External V_needed Bland-Altman")
                        st.pyplot(fig_ba)
                        st.caption(f"External V_needed bias {mb:+.1f} mL "
                                   f"(internal reference −13.7 mL).")

                # Downloadable per-donor external predictions
                out = df_ext.copy()
                out["yield_P50_pred"] = yq[:, 1]
                out["eff_P50_pred"] = eq[:, 1]
                st.download_button(
                    "⬇ Download external predictions (CSV)",
                    out.to_csv(index=False).encode(),
                    file_name="cd34_external_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Could not process external cohort: {e}")

st.markdown("---")
st.caption("CD34+ Mobilization Decision Support — multi-task TabResNet research prototype. "
           "Predictions are model estimates and must be reviewed by a qualified clinician. "
           "Not a medical device.")
