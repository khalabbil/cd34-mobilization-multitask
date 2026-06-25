# Multi-task TabResNet for CD34+ mobilization prediction

Analysis code for the manuscript **"Calibrated prediction of CD34+ yield and collection efficiency for allogeneic donor apheresis product-volume planning."**

A multi-task tabular residual neural network (TabResNet) that jointly predicts, from pre-apheresis donor features:

1. **CD34+ yield** (×10⁶ CD34+/kg recipient weight), and
2. **Collection efficiency** (×10⁶ CD34+/mL apheresis product) — a recipient-weight–independent endpoint,

each with **calibrated 80% prediction intervals** via quantile heads, and derives a **V_needed nomogram** that converts a donor's features and a target CD34+ dose into the required apheresis product volume with an explicit interval.

## Highlights

- Multi-task TabResNet: feature-tokenizer embeddings → shared residual trunk → two quantile heads (P10/P50/P90), monotonicity enforced by construction.
- Calibrated 80% prediction intervals (prediction-interval coverage probability, PICP-80, within the prespecified 0.75–0.85 band).
- V_needed apheresis product-volume nomogram for collection planning.
- Fair benchmarking against a target-tuned quantile-XGBoost, default XGBoost, ridge regression, and a multilayer perceptron.
- 10×5 repeated stratified cross-validation with five-seed ensembling; donor-level bootstrap confidence intervals.
- Reporting aligned with TRIPOD+AI 2024.

## ⚠️ Data availability

**This repository contains source code only.** The patient-level (healthy-donor) data are **not** included and are **not** publicly available, because they contain potentially identifying information and are subject to institutional and national data-protection regulations (Turkish Personal Data Protection Law, KVKK). De-identified data may be made available from the corresponding author on reasonable request, subject to approval by the Ethics Committee of Bursa Yüksek İhtisas Training and Research Hospital and a data-sharing agreement.

To run the pipeline you must supply your own data file at `data/donor_data.xlsx` matching the expected column schema (see `configs/v4_config.yaml` for the feature list).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.12; PyTorch, scikit-learn, XGBoost, SHAP. For deterministic XGBoost behavior on some platforms, set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.

## Usage

```bash
# Smoke test (data loading + model build)
python run_pipeline.py --config configs/v4_config.yaml --mode smoke

# Full training — Model A (pre-G-CSF) and Model B (pre-apheresis CD34+)
python run_pipeline.py --config configs/v4_config.yaml --mode train

# Calibration, SHAP, sensitivity, and figures
python run_pipeline.py --config configs/v4_config.yaml --mode report

# Single-donor inference (provide your own donor feature JSON)
python run_pipeline.py --config configs/v4_config.yaml --mode predict --donor your_donor.json
```

Inference returns, for each donor, the P10/P50/P90 of yield and efficiency, the binary ≥ 5 ×10⁶/kg risk flag, and a V_needed table (product volume to reach each target dose, with 80% intervals).

An interactive Streamlit decision-support app (single-donor, batch, internal-validation, and external-validation tabs) is in `app/`:

```bash
streamlit run app/cd34_decision_support_app.py
```

## Repository structure

```
src/
  data.py, preprocessing.py        # loading, feature engineering, imputation, scaling
  model.py, losses.py              # multi-task TabResNet + pinball/quantile loss
  train.py, crossval.py            # training loop + 10×5 repeated stratified CV
  baselines.py, quantile_xgb_baseline.py   # XGBoost, ridge, MLP, tuned quantile-XGBoost
  calibration.py, shap_analysis.py, sensitivity.py, inference.py, report.py
run_pipeline.py                    # orchestrator (smoke/train/report/predict/sensitivity/full)
make_tct_figures.py                # publication-quality figure generation
app/cd34_decision_support_app.py   # Streamlit decision-support app
configs/v4_config.yaml             # feature list, model and training hyperparameters
```

## Reporting & ethics

Single-centre, retrospective study; approved by the Ethics Committee of Bursa Yüksek İhtisas Training and Research Hospital (approval number 2024-TBEK 2025/04-01). Internal validation only — external, prospective, multicentre validation is required before clinical use. **For research use only; not a medical device.**

## Citation

If you use this code, please cite the manuscript (citation to be completed upon publication):

> Gözden HE, Orhan B, Alkış N. Calibrated prediction of CD34+ yield and collection efficiency for allogeneic donor apheresis product-volume planning. *Transplantation and Cellular Therapy.* (under review).

## License

Released under the MIT License — see [LICENSE](LICENSE).
