# 🛡️ OASIS Security — Crime & Delinquency Analysis in France

> **CDSD Certification Project — RNCP35288**  
> Data Science Designer & Developer

[![Hugging Face](https://img.shields.io/badge/🤗%20HF%20Space-Live%20Demo-blue)](https://huggingface.co/spaces/Dreipfelt/oasis-security)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-0194E2?logo=mlflow&logoColor=white)](./mlruns)
[![Docker](https://img.shields.io/badge/Docker-multi--stage-2496ED?logo=docker&logoColor=white)](./models/crime_predictor/Dockerfile)
[![License](https://img.shields.io/badge/Data-data.gouv.fr-green)](https://www.data.gouv.fr)

---

## 📌 Context & Business Problem

Recorded crime and delinquency data in France is publicly available but rarely
surfaced in an accessible, analytical format. Law enforcement agencies, local
authorities, and researchers require tools to identify trends, compare regions,
and anticipate future developments.

**OASIS Security** addresses this gap by delivering a complete, production-grade
data science pipeline — from raw government CSV to interactive forecasting
dashboard and REST inference API — covering all 18 administrative regions of
metropolitan and overseas France.

**Key question:**
> *Can we accurately model and forecast regional crime trends in France from
> 2016 to 2030 using recorded Police Nationale and Gendarmerie Nationale
> statistics?*

**Answer:** Yes — our best model (Gradient Boosting) achieves **R² = 0.979**
on the held-out test set, with a cross-validated R² of **0.978 ± 0.002**,
confirming strong generalisation.

---

## 🏆 Model Performance Summary

| Model | R² Test | RMSE Test | MAE Test | CV R² Mean | CV R² Std |
|---|---|---|---|---|---|
| **Gradient Boosting** ✅ | **0.9793** | **48.84** | **29.95** | **0.9777** | **0.0022** |
| XGBoost | 0.9781 | 50.21 | 30.90 | 0.9766 | 0.0028 |
| Random Forest | 0.9724 | 56.33 | 39.72 | 0.9684 | 0.0026 |
| Ridge | 0.0218 | 335.48 | 249.28 | 0.0065 | 0.0458 |

> All experiments tracked with **MLflow** — see `mlruns/` for full run history,
> parameters, and artefacts.

---

## 🗂️ Dataset

| Property | Details |
|---|---|
| **Source** | [data.gouv.fr](https://www.data.gouv.fr) |
| **Publisher** | Police Nationale & Gendarmerie Nationale |
| **Scope** | All 18 French administrative regions (INSEE 2025) |
| **Period** | 2016–2025 |
| **Granularity** | Region × Crime category × Year |
| **Format** | CSV (semicolon-delimited, UTF-8) |
| **Update frequency** | Annual |

The dataset is loaded dynamically at runtime from its canonical URL on
`static.data.gouv.fr`, ensuring the application always reflects the latest
published figures without manual intervention.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      DATA PIPELINE                          │
│                                                             │
│  data.gouv.fr ──► load_data() ──► detect_columns()          │
│                                         │                   │
│                                ┌────────▼────────┐          │
│                                │  Preprocessing  │          │
│                                │  · Type casting │          │
│                                │  · Null handling│          │
│                                │  · Label mapping│          │
│                                └────────┬────────┘          │
└─────────────────────────────────────────┼───────────────────┘
                                          │
┌─────────────────────────────────────────▼───────────────────┐
│                    FEATURE ENGINEERING                      │
│                                                             │
│  · Cyclic temporal features  (year_sin, year_cos)           │
│  · Trend normalisation       (year_trend)                   │
│  · Lag features              (lag1, lag2)                   │
│  · Rolling mean              (roll_mean_3)                  │
│  · Regional aggregates       (region_mean)                  │
│  · Categorical encoding      (ind_code, reg_code)           │
└─────────────────────────────────────────┬───────────────────┘
                                          │
┌─────────────────────────────────────────▼───────────────────┐
│                    MODELLING LAYER                          │
│                                                             │
│  ┌───────────────────┐      ┌───────────────────────────┐   │
│  │  Train set        │      │  Test set (held out)      │   │
│  │  2016 → 2023      │─────►│  2024–2025                │   │
│  └─────────┬─────────┘      └───────────────────────────┘   │
│            │                                                │
│  ┌─────────▼──────────────────────────────────────────┐     │
│  │  Gradient Boosting · XGBoost · Random Forest       │     │
│  │  Ridge · LightGBM · Prophet · Holt-Winters         │     │
│  └─────────────────────────┬──────────────────────────┘     │
│                            │                                │
│          TimeSeriesSplit cross-validation (n=3)             │
│          MLflow experiment tracking (12 runs)               │
│          → Champion: Gradient Boosting (R²=0.979)           │
└─────────────────────────────────────────┬───────────────────┘
                                          │
┌─────────────────────────────────────────▼───────────────────┐
│                    SERVING LAYER                            │
│                                                             │
│  ┌────────────────────────┐   ┌────────────────────────┐    │
│  │  Streamlit Dashboard   │   │  FastAPI REST API      │    │
│  │  (Hugging Face Spaces) │   │  (Docker container)    │    │
│  │  streamlit/app.py      │   │  models/.../predict.py │    │
│  └────────────────────────┘   └────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🤖 Modelling Approach

### Problem framing
Each (region, crime category) pair forms an independent supervised regression
problem. The target variable is the annual number of recorded offences per
100,000 inhabitants (`taux_100k`).

### Feature engineering
Production-grade features are constructed for each observation:

- **Cyclic temporal encoding** — `year_sin` and `year_cos` capture periodicity
  without imposing linearity on the year variable
- **Lag features** — `lag1` and `lag2` provide the model with recent history
  per (indicator, region) group
- **Rolling mean** — `roll_mean_3` smooths short-term volatility
- **Regional aggregates** — `region_mean` contextualises each series within
  its regional baseline
- **Categorical encoding** — indicators and regions are ordinally encoded

### Validation strategy
A `TimeSeriesSplit` with 3 folds is used throughout, respecting the temporal
ordering of observations and preventing data leakage from future to past.

### Experiment tracking
All model runs are logged with **MLflow**, including:

- Hyperparameters (`model`, `n_estimators`, `learning_rate`, etc.)
- Metrics (`r2_train`, `r2_test`, `rmse_test`, `mae_test`, `cv_r2_mean`, `cv_r2_std`)
- Model artefacts (serialised `.pkl` files)
- Git commit hash for full reproducibility

---

## 🛠️ Technical Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.11 |
| Dashboard | Streamlit | 1.45 |
| Visualisation | Plotly Express & Graph Objects | ≥ 5.18 |
| Data processing | Pandas, NumPy | ≥ 2.0, ≥ 1.24 |
| ML — Boosting | LightGBM, XGBoost, GradientBoosting | ≥ 4.3, ≥ 1.7 |
| ML — Forecasting | Prophet, Statsmodels (Holt-Winters) | 1.1, ≥ 0.14 |
| ML — Utilities | Scikit-learn (TimeSeriesSplit, metrics) | ≥ 1.3 |
| Experiment tracking | MLflow | ≥ 2.12 |
| REST API | FastAPI + Uvicorn | ≥ 0.110 |
| Containerisation | Docker (multi-stage build) | — |
| Deployment | Hugging Face Spaces (Streamlit SDK) | — |

---

## 🐳 MLOps & Containerisation

The inference pipeline is fully containerised using a **multi-stage Docker
build**, cleanly separating the training environment from the production image.

```
Stage 1 — trainer
  · Installs full ML stack (LightGBM, XGBoost, Prophet, statsmodels…)
  · Receives DATA_URL as a build argument
  · Runs train.py → serialises crime_predictor.pkl

Stage 2 — production
  · Copies only the serialised artefact from Stage 1
  · Installs minimal serving dependencies (fastapi, uvicorn, pandas, numpy)
  · Exposes port 8000 with HEALTHCHECK
  · Runs as non-root user (security best practice)
```

```bash
# Build
docker build \
  --build-arg DATA_URL="https://static.data.gouv.fr/.../donnee-reg.csv" \
  -t oasis-security:latest \
  ./models/crime_predictor/

# Run
docker run -p 8000:8000 oasis-security:latest

# Health check
curl http://localhost:8000/health

# Inference
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"region": "11", "crime_category": "Vols avec violence", "horizon": 5}'
```

---

## 📁 Repository Structure

```
oasis-security/
│
├── README.md                          # This file
├── LICENSE
├── .gitignore
├── requirements.txt                   # Top-level dependencies
├── Dockerfile                         # Root-level compose target
├── docker-compose.yml
│
├── data/
│   ├── raw/                           # Source files (never modified)
│   ├── processed/                     # Cleaned, model-ready CSVs
│   ├── geo/                           # Geospatial files (GeoJSON)
│   └── docs/                          # Dataset documentation
│
├── notebooks/
│   ├── 01_exploration_crimes.ipynb    # Data exploration & EDA
│   ├── 02_benchmark_modeles.ipynb     # Model comparison & selection
│   └── 03_analyse_departements.ipynb  # Departmental deep-dive
│
├── pipeline/                          # Reusable data pipeline modules
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   └── crime_predictor/
│       ├── Dockerfile                 # Multi-stage build (train → serve)
│       ├── artifacts/
│       │   ├── crime_predictor.pkl    # Serialised champion model
│       │   └── metrics.json          # Benchmark results (R²=0.979)
│       ├── src/
│       │   ├── config.yaml           # Hyperparameters & data config
│       │   ├── model.py              # CrimeRatePredictor class
│       │   ├── train.py              # Training pipeline
│       │   └── predict.py            # FastAPI inference endpoint
│       └── tests/
│           └── test_model.py
│
├── mlruns/                            # MLflow tracking (12 runs logged)
│
├── images/                            # Visuals for documentation
│
└── streamlit/                         # Hugging Face Space
    ├── app.py
    └── requirements.txt
```

---

## 🚀 Running Locally

### Dashboard

```bash
git clone https://github.com/Data-Science-Designer-and-Developer/oasis-security.git
cd oasis-security

pip install -r requirements.txt
streamlit run streamlit/app.py
```

### Inference API

```bash
cd models/crime_predictor

docker build \
  --build-arg DATA_URL="https://static.data.gouv.fr/.../donnee-reg.csv" \
  -t oasis-security:latest .

docker run -p 8000:8000 oasis-security:latest
```

### MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns
# Open http://localhost:5000
```

---

## ⚖️ Ethics & Data Privacy

The data used throughout this project is:

- **Publicly available** — published by French government authorities under
  Licence Ouverte v2.0
- **Aggregated** — figures are presented at regional level only; no
  individual-level records are processed or stored
- **Non-identifiable** — no re-identification of persons is possible from
  the published aggregates

This project is intended solely for informational, educational, and analytical
purposes. Forecasts are indicative and subject to the inherent limitations of
statistical modelling on short time series. The analysis carries no
discriminatory intent with respect to geographical areas or populations.

Data processing complies with the principles of the **GDPR** (Regulation (EU)
2016/679), in particular data minimisation, purpose limitation, and storage
limitation.

> ⚠️ Recorded crime figures reflect offences *registered* by police and
> gendarmerie services — not actual crime rates. Under-reporting, changes in
> classification practices, and variations in policing intensity may all
> influence the figures independently of true crime levels.

---

## 📜 Licence

Data: [Licence Ouverte v2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence)
— © Police Nationale & Gendarmerie Nationale / data.gouv.fr

Code: MIT


--- 
## Author
Frédéric Tellier  
  LinkedIn: https://www.linkedin.com/in/frédéric-tellier-8a9170283/  
  Portfolio: https://github.com/Dreipfelt  

---

*CDSD Certification Project — Data Science Designer & Developer (RNCP35288)*
