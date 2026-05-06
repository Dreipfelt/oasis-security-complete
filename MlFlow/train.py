# MlFlow/train.py
# Training pipeline with MLflow tracking
# Artefacts pushed to hf://datasets/Dreipfelt/oasis-mlflow-artifacts

import os
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

# ── MLflow remote tracking server (HF Space) ──────────────
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "https://dreipfelt-oasis-mlflow.hf.space"
)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("oasis-security-crime-prediction")

# ── Data source ────────────────────────────────────────────
DATA_URL = os.getenv(
    "DATA_URL",
    "https://static.data.gouv.fr/resources/"
    "bases-statistiques-communale-departementale-et-regionale-de-la-delinquance-"
    "enregistree-par-la-police-et-la-gendarmerie-nationales/"
    "20260129-160256/donnee-reg-data.gouv-2025-geographie2025-produit-le2026-01-22.csv"
)

# ── Models to benchmark ────────────────────────────────────
MODELS = {
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.08,
        random_state=42
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=200, max_depth=6, random_state=42, n_jobs=-1
    ),
    "Ridge": Ridge(alpha=1.0),
}


def load_data(url: str) -> pd.DataFrame:
    """Load and clean data from data.gouv.fr"""
    print(f"📥 Loading data from {url[:60]}...")
    df = pd.read_csv(url, sep=";", encoding="utf-8", low_memory=False)
    df = df[df["unite_de_compte"] == "nombre"].copy()
    df["taux_100k"] = df["nombre"] / df["insee_pop"] * 100_000
    return df.dropna(subset=["taux_100k"])


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Production-grade feature engineering"""
    df = df.copy()

    # Tri temporel obligatoire avant tout calcul de lag
    df = df.sort_values(["indicateur", "Code_region", "annee"]).reset_index(drop=True)

    # Cyclic temporal encoding
    df["year_sin"] = np.sin(2 * np.pi * df["annee"] / 10)
    df["year_cos"] = np.cos(2 * np.pi * df["annee"] / 10)
    df["year_trend"] = (
        (df["annee"] - df["annee"].min())
        / (df["annee"].max() - df["annee"].min())
    )

    # Lag features (per indicator × region group)
    grp = df.groupby(["indicateur", "Code_region"])["taux_100k"]
    df["lag1"] = grp.shift(1).fillna(grp.transform("mean"))
    df["lag2"] = grp.shift(2).fillna(grp.transform("mean"))
    df["roll_mean_3"] = (
        grp.rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    )

    # Regional aggregate
    df["region_mean"] = df.groupby("Code_region")["taux_100k"].transform("mean")

    # Categorical encoding
    df["ind_code"] = pd.Categorical(df["indicateur"]).codes
    df["reg_code"] = pd.Categorical(df["Code_region"]).codes

    feature_cols = [
        "year_sin", "year_cos", "year_trend",
        "lag1", "lag2", "roll_mean_3",
        "region_mean", "ind_code", "reg_code",
    ]
    return df[feature_cols + ["taux_100k"]].dropna()


def evaluate(model, X_test, y_test) -> dict:
    """Compute test set metrics. Clip predictions to 0 (business constraint)."""
    preds = model.predict(X_test)                   # ← ligne restaurée
    preds = np.clip(preds, a_min=0, a_max=None)     # taux ne peut pas être négatif
    return {
        "r2_test":   round(r2_score(y_test, preds), 4),
        "rmse_test": round(np.sqrt(mean_squared_error(y_test, preds)), 4),
        "mae_test":  round(mean_absolute_error(y_test, preds), 4),
    }


def train_and_log(model_name: str, model, X_train, X_test, y_train, y_test):
    """Train one model and log everything to MLflow"""
    print(f"\n🔧 Training {model_name}...")

    with mlflow.start_run(run_name=model_name):

        # Cross-validation (TimeSeriesSplit — no future leakage)
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=tscv, scoring="r2", n_jobs=-1
        )

        # Final fit on full train set
        model.fit(X_train, y_train)
        r2_train = model.score(X_train, y_train)

        # Test metrics
        metrics = evaluate(model, X_test, y_test)
        metrics["r2_train"]   = round(r2_train, 4)
        metrics["cv_r2_mean"] = round(cv_scores.mean(), 4)
        metrics["cv_r2_std"]  = round(cv_scores.std(), 4)

        # Log to MLflow
        mlflow.log_param("model", model_name)
        mlflow.log_metrics(metrics)

        # Log model artefact → pushed to HF Dataset
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=f"crime_predictor_{model_name.lower()}",
        )

        print(f"   R² test={metrics['r2_test']} · "
              f"RMSE={metrics['rmse_test']} · "
              f"CV R²={metrics['cv_r2_mean']}±{metrics['cv_r2_std']}")

    return metrics


def main():
    # ── Load & prepare data ────────────────────────────────
    df = load_data(DATA_URL)
    df_features = engineer_features(df)

    FEATURE_COLS = [
        "year_sin", "year_cos", "year_trend",
        "lag1", "lag2", "roll_mean_3",
        "region_mean", "ind_code", "reg_code",
    ]
    X = df_features[FEATURE_COLS]
    y = df_features["taux_100k"]

    # Temporal train/test split — last 20% as test
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print(f"📊 Train: {len(X_train)} rows · Test: {len(X_test)} rows")

    # ── Benchmark all models ───────────────────────────────
    all_metrics = {}
    for name, model in MODELS.items():
        all_metrics[name] = train_and_log(
            name, model, X_train, X_test, y_train, y_test
        )

    # ── Select champion ────────────────────────────────────
    best_name = max(all_metrics, key=lambda k: all_metrics[k]["r2_test"])
    best_metrics = all_metrics[best_name]
    print(f"\n🏆 Champion: {best_name} (R²={best_metrics['r2_test']})")

    # ── Retrain champion on full dataset ──────────────────
    champion = MODELS[best_name]
    champion.fit(X, y)

    artifacts_dir = Path(__file__).parent.parent / "models" / "crime_predictor" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(champion, artifacts_dir / "crime_predictor.pkl")

    metrics_out = {"best_model": best_name, **best_metrics, "all_models": all_metrics}
    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"✅ Artefacts saved to {artifacts_dir}")
    print(f"📊 metrics.json: R²={best_metrics['r2_test']}")


if __name__ == "__main__":
    main()
