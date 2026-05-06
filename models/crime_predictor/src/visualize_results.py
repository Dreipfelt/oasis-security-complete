# visualize_results.py
# =============================================================================
# Génère des visualisations pour la soutenance.
# Utilisation : python visualize_results.py --model-path "artifacts/crime_predictor.pkl"
#
# Corrections :
#   - plot_feature_importance() : fallback gracieux si le modèle (ex: LinearRegression)
#     n'a pas feature_importances_ (AttributeError silencieux dans l'original)
#   - Titre du graphique dynamique (nom du champion, pas "LightGBM" hardcodé)
#   - load() est maintenant une classmethod → CrimeRatePredictor.load(path) ✓
# =============================================================================

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parent))
from model import CrimeRatePredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def plot_feature_importance(predictor: CrimeRatePredictor, output_dir: str = "plots") -> None:
    """Trace l'importance des features du modèle champion.

    Si le modèle ne supporte pas feature_importances_ (ex: LinearRegression),
    trace les coefficients normalisés à la place.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "feature_importance.png"

    if predictor.model is None:
        logger.error("Aucun modèle chargé.")
        return

    model_name = predictor.best_model_name or "Modèle"

    # CORRECTION : fallback sur coef_ si feature_importances_ absent
    if hasattr(predictor.model, "feature_importances_"):
        importance = predictor.model.feature_importances_
        importance_label = "Importance"
    elif hasattr(predictor.model, "coef_"):
        importance = abs(predictor.model.coef_)
        importance_label = "|Coefficient|"
        logger.info("%s : pas de feature_importances_, utilisation de |coef_|.", model_name)
    else:
        logger.warning("%s ne supporte ni feature_importances_ ni coef_. Graphique ignoré.", model_name)
        return

    feature_names = getattr(predictor.model, "feature_names_in_", None)

    if feature_names is not None:
        feature_names = list(feature_names)
    else:
        feature_names = list(predictor.feature_names or predictor.FEATURE_COLS)

    if len(feature_names) != len(importance):
        logger.warning(
            "Nombre de features différent : %d noms vs %d importances. "
            "Noms génériques utilisés.",
            len(feature_names),
            len(importance),
        )
        feature_names = [f"feature_{i}" for i in range(len(importance))]

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    plt.figure()
    sns.barplot(x="importance", y="feature", data=importance_df, palette="viridis")
    plt.title(f"Importance des Features ({model_name})")
    plt.xlabel(importance_label)
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Graphique sauvegardé : %s", output_path)


def plot_predictions_vs_actual(
    csv_path: str = "test_predictions.csv",
    output_dir: str = "plots",
) -> None:
    """Trace les prédictions vs les valeurs réelles depuis un CSV."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "predictions_vs_actual.png"

    try:
        df = pd.read_csv(csv_path)
        if "y_true" not in df.columns or "y_pred" not in df.columns:
            logger.error("Le CSV doit contenir les colonnes 'y_true' et 'y_pred'.")
            return

        vmin = min(df["y_true"].min(), df["y_pred"].min())
        vmax = max(df["y_true"].max(), df["y_pred"].max())

        plt.figure()
        sns.scatterplot(x="y_true", y="y_pred", data=df, alpha=0.5, label="Prédictions")
        plt.plot([vmin, vmax], [vmin, vmax], "r--", label="Ligne idéale (y=x)")
        plt.xlabel("Valeurs réelles (taux pour 100k habitants)")
        plt.ylabel("Prédictions")
        plt.title("Prédictions vs Valeurs Réelles")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info("Graphique sauvegardé : %s", output_path)
    except FileNotFoundError:
        logger.warning("Fichier '%s' introuvable — graphique ignoré.", csv_path)
    except Exception as exc:
        logger.error("Erreur lors du traçage des prédictions : %s", exc)


def plot_metrics_comparison(
    csv_path: str = "model_comparison.csv",
    output_dir: str = "plots",
) -> None:
    """Trace la comparaison R² entre modèles depuis un CSV."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "metrics_comparison.png"

    try:
        df = pd.read_csv(csv_path)
        if "model" not in df.columns or "r2_score" not in df.columns:
            logger.error("Le CSV doit contenir les colonnes 'model' et 'r2_score'.")
            return

        df = df.sort_values("r2_score", ascending=False)

        plt.figure()
        sns.barplot(x="model", y="r2_score", data=df, palette="viridis")
        plt.title("Comparaison des Modèles — R² Score")
        plt.xlabel("Modèle")
        plt.ylabel("R² Score")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info("Graphique sauvegardé : %s", output_path)
    except FileNotFoundError:
        logger.warning("Fichier '%s' introuvable — graphique ignoré.", csv_path)
    except Exception as exc:
        logger.error("Erreur lors du traçage de la comparaison : %s", exc)


def main():
    parser = argparse.ArgumentParser(
        description="Génère des visualisations pour la soutenance CDSD Bloc 6."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/crime_predictor/artifacts/crime_predictor.pkl",
        help="Chemin vers le fichier pickle du modèle.",
    )
    parser.add_argument(
        "--predictions-csv",
        type=str,
        default="test_predictions.csv",
        help="CSV contenant y_true et y_pred (optionnel).",
    )
    parser.add_argument(
        "--comparison-csv",
        type=str,
        default="model_comparison.csv",
        help="CSV issu de compare_models.py (optionnel).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Dossier de sortie des graphiques (défaut : plots/).",
    )
    args = parser.parse_args()

    try:
        # CORRECTION : load() est une classmethod
        predictor = CrimeRatePredictor.load(args.model_path)
        logger.info("Modèle chargé : %s", predictor.best_model_name)

        plot_feature_importance(predictor, args.output_dir)
        plot_predictions_vs_actual(args.predictions_csv, args.output_dir)
        plot_metrics_comparison(args.comparison_csv, args.output_dir)

        logger.info("✅ Visualisations générées dans '%s/'.", args.output_dir)

    except FileNotFoundError as exc:
        logger.error("❌ Fichier modèle introuvable : %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("❌ Erreur : %s", exc)
        raise


if __name__ == "__main__":
    main()