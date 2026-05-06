# tests/test_model.py
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Ajoute le chemin vers src/ pour importer model
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model import CrimeRatePredictor

# Fixture pour des données d'exemple réalistes
@pytest.fixture
def sample_data():
    """Crée un DataFrame réaliste pour les tests."""
    return pd.DataFrame({
        "annee": [2020, 2021, 2022, 2020, 2021, 2022],
        "indicateur": ["Vol", "Vol", "Vol", "Agression", "Agression", "Agression"],
        "CODGEO_2025": ["01001", "01001", "01001", "02001", "02001", "02001"],
        "nombre": [100, 120, 80, 50, 60, 70],
        "insee_pop": [100000, 100000, 100000, 200000, 200000, 200000],
        "unite_de_compte": ["Infraction"] * 6,
        "taux_pour_mille": [1.0, 1.2, 0.8, 0.25, 0.3, 0.35]
    })

@pytest.fixture
def predictor(sample_data):
    """Initialise un prédicteur."""
    return CrimeRatePredictor()

def test_load_data(tmp_path, sample_data):
    """Teste la fonction load_data avec un fichier temporaire."""
    parquet_path = tmp_path / "test_data.parquet"
    sample_data.to_parquet(parquet_path)
    predictor = CrimeRatePredictor()
    df = predictor.load_data(str(parquet_path))
    assert "taux_100k" in df.columns
    assert len(df) == len(sample_data)

def test_engineer_features(sample_data):
    """Teste la fonction engineer_features."""
    predictor = CrimeRatePredictor()
    df = predictor.engineer_features(sample_data)
    expected_cols = predictor.FEATURE_COLS + ["annee", "indicateur", "Code_region", "taux_100k"]
    for col in expected_cols:
        assert col in df.columns, f"Colonne manquante: {col}"

def test_train(tmp_path):
    """Teste la fonction train."""
    np.random.seed(42)
    n_samples = 500
    data = {
        "annee": np.random.randint(2015, 2025, n_samples),
        "indicateur": np.random.choice(["Vol", "Agression", "Cambriolage"], n_samples),
        "CODGEO_2025": np.random.choice(["01001", "02001", "03001", "2A001", "2B001"], n_samples),
        "nombre": np.random.randint(10, 1000, n_samples),
        "insee_pop": np.random.randint(50000, 500000, n_samples),
        "unite_de_compte": ["Infraction"] * n_samples,
        "taux_pour_mille": np.random.uniform(0.1, 10.0, n_samples)
    }
    df = pd.DataFrame(data)
    parquet_path = tmp_path / "train_data.parquet"
    df.to_parquet(parquet_path)
    predictor = CrimeRatePredictor()
    metrics = predictor.train(str(parquet_path), n_cv_splits=2, test_size=0.2)
    assert predictor.model is not None
    assert predictor.best_model_name is not None
    assert 0 <= metrics["r2_score"] <= 1

def test_predict_2030(tmp_path):
    """Teste la fonction predict_2030."""
    np.random.seed(42)
    n_samples = 500
    data = {
        "annee": np.random.randint(2015, 2025, n_samples),
        "indicateur": np.random.choice(["Vol", "Agression"], n_samples),
        "CODGEO_2025": np.random.choice(["01001", "02001"], n_samples),
        "nombre": np.random.randint(10, 1000, n_samples),
        "insee_pop": np.random.randint(50000, 500000, n_samples),
        "unite_de_compte": ["Infraction"] * n_samples,
        "taux_pour_mille": np.random.uniform(0.1, 10.0, n_samples)
    }
    df = pd.DataFrame(data)
    parquet_path = tmp_path / "predict_data.parquet"
    df.to_parquet(parquet_path)
    predictor = CrimeRatePredictor()
    predictor.train(str(parquet_path), n_cv_splits=2, test_size=0.2)
    prediction = predictor.predict_2030("Vol", "01", df)
    assert not pd.isna(prediction)
    assert isinstance(prediction, (float, np.floating))

def test_save_load(tmp_path):
    """Teste les fonctions save et load."""
    np.random.seed(42)
    n_samples = 500
    data = {
        "annee": np.random.randint(2015, 2025, n_samples),
        "indicateur": np.random.choice(["Vol", "Agression"], n_samples),
        "CODGEO_2025": np.random.choice(["01001", "02001"], n_samples),
        "nombre": np.random.randint(10, 1000, n_samples),
        "insee_pop": np.random.randint(50000, 500000, n_samples),
        "unite_de_compte": ["Infraction"] * n_samples,
        "taux_pour_mille": np.random.uniform(0.1, 10.0, n_samples)
    }
    df = pd.DataFrame(data)
    parquet_path = tmp_path / "save_load_data.parquet"
    df.to_parquet(parquet_path)
    predictor = CrimeRatePredictor()
    predictor.train(str(parquet_path), n_cv_splits=2, test_size=0.2)
    model_path = tmp_path / "test_model.pkl"
    predictor.save(model_path)
    assert model_path.exists()
    loaded_predictor = CrimeRatePredictor.load(model_path)
    assert loaded_predictor.model is not None
    assert loaded_predictor.best_model_name == predictor.best_model_name