# predict.py
# =============================================================================
# Script de prédiction en ligne de commande.
# Utilisation: python predict.py --model-path "models/crime_predictor.pkl" --indicateur "Vol" --region "1"
# =============================================================================

import argparse
import logging
import pandas as pd
from model import CrimeRatePredictor

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Prédiction du taux de criminalité pour 2030.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Chemin vers le fichier du modèle sauvegardé."
    )
    parser.add_argument(
        "--data-url",
        type=str,
        required=True,
        help="URL ou chemin vers le fichier CSV des données historiques."
    )
    parser.add_argument(
        "--indicateur",
        type=str,
        required=True,
        help="Type de criminalité (ex: 'Vol', 'Agression')."
    )
    parser.add_argument(
        "--region",
        type=str,
        required=True,
        help="Code de la région (ex: '1' pour Île-de-France)."
    )
    args = parser.parse_args()

    try:
        # Charger le modèle
        predictor = CrimeRatePredictor.load(args.model_path)

        # Charger les données historiques
        df = predictor.load_data(args.data_url)

        # Faire la prédiction
        prediction = predictor.predict_2030(
        indicateur=args.indicateur,
        code_region=args.region,
        df_history=df
        )

        if pd.isna(prediction):
            logger.error("Aucune prédiction possible (données manquantes).")
        else:
            logger.info(f"Prédiction pour {args.indicateur} (région {args.region}) en 2030: {prediction:.2f} pour 100k habitants")

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise

if __name__ == "__main__":
    main()