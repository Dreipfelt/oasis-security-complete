# OASIS Security — CDSD Bloc 6

## Project Overview

OASIS Security is a Data Science and MLOps project designed to analyse and predict the evolution of crime and offence indicators recorded in France using official public datasets published on data.gouv.fr.

The project covers:

* data ingestion and preprocessing,
* temporal feature engineering,
* comparison of multiple Machine Learning models,
* training and serialisation of the champion model,
* generation of analytical visualisations,
* CI/CD automation,
* deployment of a Streamlit dashboard.

---

# Objectives

* Analyse the evolution of crime indicators by region.
* Build a robust prediction pipeline.
* Compare multiple regression models.
* Implement a reproducible MLOps architecture.
* Provide meaningful visualisations for business analysis.

---

# Project Structure

```text
oasis-security/
│
├── .github/workflows/       # GitHub Actions CI/CD
├── data/                    # Processed and geographical datasets
├── docs/                    # GitHub Pages documentation
├── images/                  # Exported visualisations
├── MlFlow/                  # MLflow tracking
├── models/
│   └── crime_predictor/
│       ├── artifacts/       # Serialised models
│       ├── src/             # ML source code
│       └── tests/           # Pytest test suite
├── notebooks/               # Exploratory analysis
├── pipeline/                # Training pipeline
├── streamlit/               # Streamlit dashboard
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

# Data Sources

Official sources:

* data.gouv.fr
* French Ministry of the Interior
* INSEE

Dataset categories:

* recorded crimes and offences,
* regional datasets,
* administrative geographical data,
* historical time-series data since 2012.

---

# Feature Engineering

The engineered features include:

* cyclical temporal variables,
* yearly trend variables,
* temporal lag features,
* rolling averages,
* regional encoding,
* offence category encoding.

Examples:

* `lag1`
* `lag2`
* `roll_mean_3`
* `year_sin`
* `year_cos`
* `region_mean`

---

# Compared Models

The benchmark includes:

* LinearRegression
* GradientBoostingRegressor
* RandomForestRegressor
* XGBoostRegressor
* LightGBMRegressor

---

# Results

## Final Benchmark

| Model            | Test R² |    RMSE |    MAE |
| ---------------- | ------: | ------: | -----: |
| LinearRegression |  0.9990 |  1.8852 | 0.0445 |
| GradientBoosting |  0.9823 |  7.8544 | 1.3526 |
| RandomForest     |  0.9638 | 11.2344 | 1.1232 |
| LightGBM         |  0.9207 | 16.6169 | 3.8849 |
| XGBoost          |  0.8917 | 19.4267 | 4.1613 |

---

# Performance Analysis

The `LinearRegression` model achieved the highest raw metrics (`R² ≈ 0.999`), mainly due to the strong correlation between temporal lag variables (`lag1`, `lag2`, `rolling mean`) and the target variable.

To avoid a naïve interpretation of the results, several ensemble and boosting models were also evaluated:

* Gradient Boosting,
* Random Forest,
* XGBoost,
* LightGBM.

This approach makes it possible to:

* compare model robustness,
* reduce the risk of overfitting,
* verify performance stability,
* validate the consistency of the feature engineering pipeline.

---

# Machine Learning Pipeline

The complete pipeline includes:

1. Data loading
2. Data cleaning and normalisation
3. Feature engineering
4. Temporal splitting using `TimeSeriesSplit`
5. Multi-model benchmarking
6. Champion model selection
7. Model serialisation
8. Metrics generation
9. Prediction generation
10. Visualisation export

---

# MLOps Stack

Technologies used in the project:

* Python 3.11+
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM
* MLflow
* Streamlit
* Docker
* GitHub Actions
* Pytest

---

# Testing

The project includes:

* unit tests,
* integration tests,
* model serialisation tests,
* prediction consistency tests.

Final test status:

```bash
21 passed
```

---

# Visualisations

Generated visualisations include:

* regional crime heatmaps,
* feature importance analysis,
* model comparison charts,
* temporal trend analysis,
* prediction dashboards.

---

# Running the Project

## Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the training pipeline

```bash
python pipeline/train.py
```

---

## Generate the champion model

```bash
python models/crime_predictor/src/generate_model.py \
  --data-path "path/to/dataset.parquet"
```

---

## Generate predictions

```bash
python models/crime_predictor/src/predict.py \
  --model-path "../artifacts/crime_predictor.pkl" \
  --data-url "path/to/dataset.parquet" \
  --indicateur "Cambriolage" \
  --region "11"
```

---

# Reproducibility

The project was designed to ensure:

* reproducible experiments,
* isolated environments,
* tracked metrics,
* automated testing,
* consistent training pipelines.

---

# Ethical Considerations

This project uses public aggregated datasets only.

No personal or sensitive individual data is processed.

The predictions generated by the models are intended solely for educational, analytical, and research purposes.

---

# Author

Frédéric Tellier
CDSD — Bloc 6 Certification Project