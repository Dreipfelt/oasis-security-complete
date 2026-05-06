# train.py
# =============================================================================
# Script d'entraînement avec MLflow pour le suivi des expériences.
# Utilisation: python train.py --data-url "URL_DU_CSV" --experiment-name "nom_experience"
# =============================================================================

import argparse
import logging
from pathlib import Path
import mlflow
import mlflow.lightgbm
import pandas as pd
from model import CrimeRatePredictor

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    # Parsing des arguments
    parser = argparse.ArgumentParser(description="Entraînement du modèle CrimeRatePredictor.")
    parser.add_argument(
        "--data-url",
        type=str,
        required=True,
        help="URL ou chemin vers le fichier CSV des données."
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="crime_predictor",
        help="Nom de l'expérience MLflow."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="lightgbm_v1",
        help="Nom de la run MLflow."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/crime_predictor.pkl",
        help="Chemin pour sauvegarder le modèle."
    )
    parser.add_argument(
        "--n-cv-splits",
        type=int,
        default=3,
        help="Nombre de folds pour la validation croisée."
    )
    args = parser.parse_args()

    # Configuration de MLflow
    mlflow.set_experiment(args.experiment_name)
    # Pour un serveur local: mlflow.set_tracking_uri("http://localhost:5000")
    # Pour un serveur distant: mlflow.set_tracking_uri("http://ton-serveur-mlflow:5000")
    # Pour un suivi local (fichiers): mlflow.set_tracking_uri("file:///tmp/mlruns")

    with mlflow.start_run(run_name=args.run_name):
        try:
            # Initialisation et entraînement
            predictor = CrimeRatePredictor()
            metrics = predictor.train(
                data_url=args.data_url,
                n_cv_splits=args.n_cv_splits
            )

            # Logging des paramètres
            mlflow.log_params(predictor.config["model"])

            # Logging des métriques
            for metric_name, value in metrics.items():
                if metric_name != "feature_importance":
                    mlflow.log_metric(metric_name, value)

            # Logging de l'importance des features
            for feature, importance in metrics["feature_importance"].items():
                mlflow.log_metric(f"feature_importance_{feature}", importance)

            # Logging du modèle
            mlflow.lightgbm.log_model(
                predictor.model,
                "model",
                input_example=pd.DataFrame(
                    {f: [0.0] for f in predictor.FEATURE_COLS}
                ).head(1)
            )

            # Sauvegarde locale
            Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
            predictor.save(args.model_path)
            logger.info(f"✅ Modèle sauvegardé: {args.model_path}")

            # Logging des artefacts (prédictions)
            mlflow.log_artifact("test_predictions.csv")

            logger.info(f"📊 Métriques finales: {metrics}")

        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement: {e}")
            raise

if __name__ == "__main__":
    main()