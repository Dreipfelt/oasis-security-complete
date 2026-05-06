# generate_model.py
# =============================================================================
# Entraîne le modèle champion sur les données réelles et sauvegarde le pickle.
# Utilisation : python generate_model.py
#
# Corrections :
#   - PROJECT_ROOT résolu dynamiquement depuis __file__ (plus de chemin absolu
#     codé en dur — fonctionne partout après git clone)
#   - Chemin données et modèle configurables via variables d'env ou args
#   - Log des métriques complètes du champion
# =============================================================================

import argparse
import logging
import sys
from pathlib import Path

# Résolution dynamique : ce script est dans oasis-security/models/crime_predictor/src/
# ou dans oasis-security/notebooks/ selon où il est placé.
# On remonte jusqu'à la racine du projet (dossier contenant models/ et data/).
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE
for _ in range(4):  # remonte au plus 4 niveaux
    if (_PROJECT_ROOT / "models").exists() and (_PROJECT_ROOT / "data").exists():
        break
    _PROJECT_ROOT = _PROJECT_ROOT.parent

sys.path.append(str(_PROJECT_ROOT / "models" / "crime_predictor" / "src"))
from model import CrimeRatePredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Valeurs par défaut (remplaçables via --data-path et --model-path)
_DEFAULT_DATA = str(
    _PROJECT_ROOT / "data"
    / "donnee-comm-data.gouv-parquet-2025-geographie2025-produit-le2026-02-03.parquet"
)
_DEFAULT_MODEL = str(
    _PROJECT_ROOT / "models" / "crime_predictor" / "artifacts" / "crime_predictor.pkl"
)


def main():
    parser = argparse.ArgumentParser(description="Entraîne et sauvegarde le modèle champion.")
    parser.add_argument("--data-path",  default=_DEFAULT_DATA,  help="Chemin vers les données.")
    parser.add_argument("--model-path", default=_DEFAULT_MODEL, help="Chemin de sortie du pickle.")
    parser.add_argument("--cv-splits",  type=int, default=3,    help="Nombre de folds CV (défaut: 3).")
    parser.add_argument("--test-size",  type=float, default=0.2, help="Fraction test (défaut: 0.2).")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Démarrage de l'entraînement…")
        logger.info("  Données  : %s", args.data_path)
        logger.info("  Sortie   : %s", model_path)

        predictor = CrimeRatePredictor()
        metrics = predictor.train(
            args.data_path,
            n_cv_splits=args.cv_splits,
            test_size=args.test_size,
        )
        predictor.save(model_path)

        logger.info("✅ Modèle sauvegardé : %s", model_path)
        logger.info("📊 Champion  : %s", predictor.best_model_name)
        logger.info("   R² test   : %.4f", metrics["r2_score"])
        logger.info("   RMSE      : %.4f", metrics["rmse"])
        logger.info("   MAE       : %.4f", metrics["mae"])
        logger.info("   CV R²     : %.4f ± %.4f", metrics["cv_r2_mean"], metrics["cv_r2_std"])

    except FileNotFoundError as exc:
        logger.error("❌ Fichier introuvable : %s", exc)
        logger.error("   Vérifier --data-path ou la présence du fichier dans data/")
        sys.exit(1)
    except Exception as exc:
        logger.error("❌ Erreur inattendue : %s", exc)
        raise


if __name__ == "__main__":
    main()