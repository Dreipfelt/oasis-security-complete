# compare_models.py
# =============================================================================
# Compare plusieurs algorithmes de régression pour la prédiction du taux de
# criminalité.
# Utilisation: python compare_models.py --data-url "URL_OU_CHEMIN_DU_FICHIER"
#
# Correction : mean_squared_error(squared=False) supprimé en sklearn 1.4+
#              → remplacé par np.sqrt(mean_squared_error(...))
# =============================================================================

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import lightgbm as lgb

sys.path.append(str(Path(__file__).parent))
from model import CrimeRatePredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compare_models(data_url: str, test_size: float = 0.2) -> pd.DataFrame:
    """Compare plusieurs modèles de régression sur les données de délinquance.

    Parameters
    ----------
    data_url : str
        URL ou chemin local vers le fichier de données (CSV ou Parquet).
    test_size : float
        Proportion des données réservée au test (split temporel).

    Returns
    -------
    pd.DataFrame
        Tableau des métriques R², RMSE, MAE pour chaque modèle, trié par R².
    """
    predictor = CrimeRatePredictor()
    df = predictor.load_data(data_url)
    df_features = predictor.engineer_features(df)

    X = df_features[predictor.FEATURE_COLS]
    y = df_features["taux_100k"]

    # Split temporel (pas de shuffle — données temporelles)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if len(X_test) == 0:
        raise ValueError(
            f"Le jeu de test est vide avec test_size={test_size}. "
            "Vérifier la taille du dataset."
        )

    models = {
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.08,
            random_state=42, verbose=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.08,
            random_state=42,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=4, random_state=42, n_jobs=-1,
        ),
        "LinearRegression": LinearRegression(),
    }

    results = []
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2   = float(r2_score(y_test, y_pred))
            # CORRECTION : squared=False supprimé en sklearn 1.4+
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae  = float(mean_absolute_error(y_test, y_pred))

            results.append({"model": name, "r2_score": r2, "rmse": rmse, "mae": mae})
            logger.info("%s — R²: %.4f | RMSE: %.4f | MAE: %.4f", name, r2, rmse, mae)

        except Exception as exc:
            logger.error("Erreur avec %s : %s", name, exc)
            results.append({"model": name, "r2_score": np.nan, "rmse": np.nan, "mae": np.nan})

    df_results = pd.DataFrame(results).sort_values("r2_score", ascending=False)
    return df_results


def main():
    parser = argparse.ArgumentParser(
        description="Compare plusieurs modèles de régression pour la prédiction de délinquance."
    )
    parser.add_argument(
        "--data-url",
        type=str,
        required=True,
        help="URL ou chemin local vers le fichier de données (CSV ou Parquet).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion du jeu de test (défaut : 0.2).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model_comparison.csv",
        help="Fichier de sortie CSV (défaut : model_comparison.csv).",
    )
    args = parser.parse_args()

    try:
        results_df = compare_models(args.data_url, test_size=args.test_size)
        results_df.to_csv(args.output, index=False)
        logger.info("Comparaison sauvegardée dans %s", args.output)
        print("\n" + results_df.to_markdown(index=False))
    except Exception as exc:
        logger.error("Erreur lors de la comparaison : %s", exc)
        raise


if __name__ == "__main__":
    main()