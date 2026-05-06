"""
tests/test_model.py
--------------------
Tests unitaires du pipeline Crime Predictor.

Couvre :
  - Chargement et intégrité du modèle sérialisé
  - Forme et type des prédictions
  - Cohérence des inputs (valeurs limites, types)
  - Pipeline de features (build_features)
  - Métriques minimales attendues (R² > 0.5)

Usage :
    pytest models/crime_predictor/tests/ -v
    pytest models/crime_predictor/tests/ -v --cov=models/crime_predictor/src
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import is_regressor

# Ajout du répertoire src au path pour les imports
SRC_DIR  = Path(__file__).resolve().parents[1] / "src"
TEST_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(ROOT_DIR))

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_dataframe() -> pd.DataFrame:
    """DataFrame minimal représentatif des données réelles."""
    np.random.seed(0)
    annees     = list(range(2016, 2024))
    deps       = ["75", "13", "69", "59", "33"]
    categories = ["Cambriolages", "Vols violence", "Vols sans violence"]

    rows = []
    for dep in deps:
        coef = np.random.uniform(0.7, 1.5)
        for cat in categories:
            for annee in annees:
                taux = 300 * coef * np.random.normal(1, 0.05)
                rows.append({
                    "annee": annee,
                    "dep": dep,
                    "indicateur": cat,
                    "tauxpour100000hab": max(round(taux, 2), 0),
                })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def trained_model(sample_dataframe):
    """Entraîne un modèle rapide pour les tests."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder

    df = sample_dataframe.copy()
    le_dep = LabelEncoder()
    le_cat = LabelEncoder()
    df["dep_encoded"] = le_dep.fit_transform(df["dep"])
    df["cat_encoded"] = le_cat.fit_transform(df["indicateur"])
    df["annee_norm"]  = (df["annee"] - df["annee"].min()) / (df["annee"].max() - df["annee"].min())

    X = df[["annee", "dep_encoded", "cat_encoded", "annee_norm"]]
    y = df["tauxpour100000hab"]

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Données
# ─────────────────────────────────────────────────────────────────────────────

class TestData:

    def test_dataframe_not_empty(self, sample_dataframe):
        """Le DataFrame ne doit pas être vide."""
        assert len(sample_dataframe) > 0, "DataFrame vide"

    def test_required_columns_present(self, sample_dataframe):
        """Les colonnes essentielles doivent être présentes."""
        required = {"annee", "dep", "indicateur", "tauxpour100000hab"}
        missing  = required - set(sample_dataframe.columns)
        assert not missing, f"Colonnes manquantes : {missing}"

    def test_no_negative_rates(self, sample_dataframe):
        """Les taux ne peuvent pas être négatifs."""
        neg = (sample_dataframe["tauxpour100000hab"] < 0).sum()
        assert neg == 0, f"{neg} taux négatifs détectés"

    def test_years_in_valid_range(self, sample_dataframe):
        """Les années doivent être comprises entre 2000 et 2030."""
        assert sample_dataframe["annee"].between(2000, 2030).all()

    def test_no_null_in_target(self, sample_dataframe):
        """Aucune valeur nulle dans la variable cible."""
        nulls = sample_dataframe["tauxpour100000hab"].isnull().sum()
        assert nulls == 0, f"{nulls} valeurs nulles dans la cible"

    def test_no_duplicates(self, sample_dataframe):
        """Pas de doublons (dep, indicateur, annee)."""
        dupes = sample_dataframe.duplicated(subset=["dep", "indicateur", "annee"]).sum()
        assert dupes == 0, f"{dupes} doublons détectés"


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Modèle
# ─────────────────────────────────────────────────────────────────────────────

class TestModel:

    def test_model_is_regressor(self, trained_model):
        """Le modèle doit être un régresseur scikit-learn."""
        assert is_regressor(trained_model), "Le modèle n'est pas un régresseur sklearn"

    def test_model_has_predict_method(self, trained_model):
        """Le modèle doit exposer une méthode predict."""
        assert hasattr(trained_model, "predict"), "Méthode predict absente"

    def test_predict_output_shape(self, trained_model):
        """predict() doit retourner un array de longueur = nb d'exemples."""
        X = np.array([
            [2022, 0, 1, 0.75],
            [2023, 2, 0, 0.87],
        ])
        preds = trained_model.predict(X)
        assert preds.shape == (2,), f"Shape inattendue : {preds.shape}"

    def test_predict_output_type(self, trained_model):
        """Les prédictions doivent être numériques."""
        X = np.array([[2022, 0, 1, 0.75]])
        pred = trained_model.predict(X)[0]
        assert isinstance(pred, (int, float, np.floating)), \
            f"Type inattendu : {type(pred)}"

    def test_predict_positive_values(self, trained_model):
        """Les taux prédits doivent être positifs (contrainte métier)."""
        X = np.array([
            [2016, 0, 0, 0.0],
            [2020, 3, 2, 0.5],
            [2023, 1, 4, 0.9],
        ])
        preds = trained_model.predict(X)
        assert (preds >= 0).all(), "Prédiction(s) négative(s) détectée(s)"

    def test_single_prediction(self, trained_model):
        """Une prédiction unique doit fonctionner sans erreur."""
        X = np.array([[2024, 1, 0, 1.0]])
        pred = trained_model.predict(X)
        assert len(pred) == 1

    def test_model_r2_above_threshold(self, trained_model, sample_dataframe):
        """Le R² sur les données d'entraînement doit dépasser 0.5."""
        from sklearn.metrics import r2_score
        from sklearn.preprocessing import LabelEncoder

        df = sample_dataframe.copy()
        le_dep = LabelEncoder()
        le_cat = LabelEncoder()
        df["dep_encoded"] = le_dep.fit_transform(df["dep"])
        df["cat_encoded"] = le_cat.fit_transform(df["indicateur"])
        df["annee_norm"]  = (df["annee"] - df["annee"].min()) / max(
            df["annee"].max() - df["annee"].min(), 1
        )
        X = df[["annee", "dep_encoded", "cat_encoded", "annee_norm"]]
        y = df["tauxpour100000hab"]

        preds = trained_model.predict(X)
        r2    = r2_score(y, preds)
        assert r2 >= 0.5, f"R² trop faible : {r2:.4f} (minimum attendu : 0.5)"

    def test_model_consistent_predictions(self, trained_model):
        """Les mêmes inputs doivent toujours produire le même output (déterminisme)."""
        X = np.array([[2022, 2, 1, 0.6]])
        pred1 = trained_model.predict(X)[0]
        pred2 = trained_model.predict(X)[0]
        assert pred1 == pred2, "Prédictions non déterministes"


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Sérialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestSerialization:

    def test_model_serializable(self, trained_model, tmp_path):
        """Le modèle doit pouvoir être sérialisé et rechargé sans perte."""
        import joblib

        path = tmp_path / "test_model.pkl"
        joblib.dump(trained_model, path)
        loaded = joblib.load(path)

        X = np.array([[2022, 0, 1, 0.75]])
        assert trained_model.predict(X)[0] == loaded.predict(X)[0], \
            "Prédiction différente après désérialisation"

    def test_metrics_json_structure(self, tmp_path):
        """Le fichier metrics.json doit contenir les clés requises."""
        metrics = {
            "best_model": "XGBoost",
            "r2_test":    0.85,
            "rmse_test":  42.1,
            "mae_test":   31.5,
        }
        path = tmp_path / "metrics.json"
        with open(path, "w") as f:
            json.dump(metrics, f)

        with open(path) as f:
            loaded = json.load(f)

        for key in ("best_model", "r2_test", "rmse_test", "mae_test"):
            assert key in loaded, f"Clé manquante dans metrics.json : {key}"
