"""
train.py — Pipeline d'entraînement du Crime Predictor
------------------------------------------------------
Compare plusieurs modèles, log les expériences dans MLflow,
sauvegarde le meilleur modèle et exporte les métriques.

Usage :
    python pipeline/train.py
    python pipeline/train.py --data-path data/crimes_clean.parquet
"""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Chemins
# ─────────────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "crimes_clean.parquet"
MODEL_DIR    = ROOT / "models" / "crime_predictor" / "artifacts"
METRICS_PATH = MODEL_DIR / "metrics.json"
MODEL_PATH   = MODEL_DIR / "crime_predictor.pkl"
MLFLOW_URI   = os.getenv(
    "MLFLOW_TRACKING_URI",
    str(ROOT / "mlruns")
)

# ─────────────────────────────────────────────────────────────────────────────
# Modèles candidats
# ─────────────────────────────────────────────────────────────────────────────
CANDIDATES = {
    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=1.0)),
    ]),
    "RandomForest": RandomForestRegressor(
        n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        random_state=42, verbosity=0,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions
# ─────────────────────────────────────────────────────────────────────────────

def load_or_generate_data(path: Path) -> pd.DataFrame:
    """Charge le Parquet ou génère des données synthétiques."""
    if path.exists():
        df = pd.read_parquet(path)
        logger.info("Données chargées : %s — %d lignes", path, len(df))
        return df

    logger.warning("Fichier absent : %s — génération de données synthétiques.", path)
    np.random.seed(42)
    annees     = list(range(2016, 2024))
    deps       = [str(i).zfill(2) for i in range(1, 21)]
    categories = ["Cambriolages", "Vols violence", "Vols sans violence",
                  "Coups blessures", "Escroqueries"]
    base       = {"Cambriolages": 280, "Vols violence": 185,
                  "Vols sans violence": 950, "Coups blessures": 520, "Escroqueries": 380}
    trend      = {"Cambriolages": -0.04, "Vols violence": 0.01,
                  "Vols sans violence": -0.02, "Coups blessures": 0.03, "Escroqueries": 0.05}
    rows = []
    for dep in deps:
        coef = np.random.uniform(0.5, 1.9)
        for cat in categories:
            for i, annee in enumerate(annees):
                taux = base[cat] * coef * (1 + trend[cat])**i * np.random.normal(1, 0.04)
                rows.append({"annee": annee, "dep": dep, "indicateur": cat,
                              "tauxpour100000hab": max(round(taux, 2), 0)})
    df = pd.DataFrame(rows)
    # Tri temporel obligatoire avant le split
    return df.sort_values(["dep", "indicateur", "annee"]).reset_index(drop=True)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Features :
        - annee (int)
        - dep_encoded  (label encoding du département)
        - cat_encoded  (label encoding de la catégorie)
        - annee_norm   (normalisé [0, 1])

    Cible :
        - tauxpour100000hab
    """
    df = df.copy().sort_values(["dep", "indicateur", "annee"]).reset_index(drop=True)

    le_dep = LabelEncoder()
    le_cat = LabelEncoder()

    df["dep_encoded"] = le_dep.fit_transform(df["dep"].astype(str))
    df["cat_encoded"] = le_cat.fit_transform(df["indicateur"].astype(str))
    df["annee_norm"]  = (df["annee"] - df["annee"].min()) / max(
        df["annee"].max() - df["annee"].min(), 1
    )

    FEATURES = ["annee", "dep_encoded", "cat_encoded", "annee_norm"]
    X = df[FEATURES]
    y = df["tauxpour100000hab"]
    return X, y


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Calcule R², RMSE, MAE sur le jeu de test. Clip à 0 (contrainte métier)."""
    preds = model.predict(X_test)                       # ← ligne restaurée
    preds = np.clip(preds, a_min=0, a_max=None)         # taux ne peut pas être négatif
    return {
        "r2_test":   round(float(r2_score(y_test, preds)),                   4),
        "rmse_test": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
        "mae_test":  round(float(mean_absolute_error(y_test, preds)),         4),
    }


def cross_validate_model(model, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> dict:
    """Cross-validation TimeSeriesSplit — jamais d'entraînement sur données futures."""
    tscv = TimeSeriesSplit(n_splits=cv)
    r2_scores = cross_val_score(model, X, y, cv=tscv, scoring="r2")
    return {
        "cv_r2_mean": round(float(r2_scores.mean()), 4),
        "cv_r2_std":  round(float(r2_scores.std()),  4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(data_path: Path) -> None:
    logger.info("=== Pipeline d'entraînement — Oasis Security ===")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Données
    df = load_or_generate_data(data_path)
    X, y = build_features(df)

    # Split temporel : les 20% dernières observations servent de test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    logger.info("Train : %d | Test : %d", len(X_train), len(X_test))

    # 2. MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("oasis-security-crime-predictor")

    best_model  = None
    best_name   = ""
    best_r2     = -np.inf
    all_results = {}

    # 3. Comparaison des modèles
    logger.info("Comparaison des modèles candidats...")
    for name, candidate in CANDIDATES.items():
        with mlflow.start_run(run_name=name):
            # Entraînement
            candidate.fit(X_train, y_train)

            # Métriques test
            metrics_test = evaluate(candidate, X_test, y_test)

            # Cross-validation
            cv_metrics = cross_validate_model(candidate, X_train, y_train, cv=5)

            # Métriques train (pour détecter overfitting)
            metrics_train = evaluate(candidate, X_train, y_train)

            all_metrics = {**metrics_test, **cv_metrics,
                           "r2_train": metrics_train["r2_test"]}
            all_results[name] = all_metrics

            # Log MLflow
            mlflow.log_params({"model": name})
            mlflow.log_metrics(all_metrics)
            mlflow.sklearn.log_model(candidate, artifact_path="model")

            logger.info(
                "%-20s | R²_test=%.4f | RMSE=%.2f | MAE=%.2f | CV_R²=%.4f±%.4f",
                name,
                metrics_test["r2_test"],
                metrics_test["rmse_test"],
                metrics_test["mae_test"],
                cv_metrics["cv_r2_mean"],
                cv_metrics["cv_r2_std"],
            )

            # Sélection du meilleur modèle (critère : R² test)
            if metrics_test["r2_test"] > best_r2:
                best_r2    = metrics_test["r2_test"]
                best_model = candidate
                best_name  = name

    # 4. Sauvegarde du meilleur modèle
    logger.info("Meilleur modèle : %s (R²_test=%.4f)", best_name, best_r2)
    joblib.dump(best_model, MODEL_PATH)
    logger.info("Modèle sauvegardé : %s", MODEL_PATH)

    # 5. Export des métriques
    best_metrics = {
        "best_model": best_name,
        **all_results[best_name],
        "all_models": all_results,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2, ensure_ascii=False)
    logger.info("Métriques exportées : %s", METRICS_PATH)

    # 6. Rapport de comparaison
    print("\n" + "=" * 65)
    print(f"{'Modèle':<22} {'R²_test':>8} {'RMSE':>8} {'MAE':>8} {'CV_R²':>10}")
    print("-" * 65)
    for name, m in all_results.items():
        marker = " ← BEST" if name == best_name else ""
        print(
            f"{name:<22} {m['r2_test']:>8.4f} {m['rmse_test']:>8.2f} "
            f"{m['mae_test']:>8.2f} {m['cv_r2_mean']:>6.4f}±{m['cv_r2_std']:.4f}{marker}"
        )
    print("=" * 65)

    logger.info("=== Entraînement terminé ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Crime Predictor")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA,
        help="Chemin vers crimes_clean.parquet",
    )
    args = parser.parse_args()
    main(args.data_path)
