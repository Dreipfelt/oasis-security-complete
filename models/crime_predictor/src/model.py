# models/crime_predictor/src/model.py
# =============================================================================
# CrimeRatePredictor — Multi-modèles, production-ready
# Compatible : Python 3.11+, pandas 2.2+, lightgbm 4+, scikit-learn 1.4+
#
# Corrections appliquées :
#   - Indentations uniformes 4 espaces (PEP 8)
#   - fillna(method='bfill') → .bfill()              (pandas >= 2.2)
#   - Tri temporel avant les lags                    (correctness)
#   - Regex code région corrigée ($ non échappé)     (bug silencieux)
#   - train() accepte test_size                      (attendu par les tests)
#   - engineer_features() retourne annee/indicateur/Code_region (tests)
#   - predict_2030() ajouté                          (attendu par les tests)
#   - load() convertie en classmethod                (attendu par les tests)
#   - train() retourne r2_score (clé exacte)         (attendu par les tests)
#   - hash() → pd.Categorical déterministe           (multi-process safety)
# =============================================================================

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapping indicateurs → catégories agrégées
# ---------------------------------------------------------------------------
INDICATEUR_MAPPING: Dict[str, str] = {
    "Vols avec armes":                                "Vol",
    "Vols violents sans arme":                        "Vol",
    "Vols dans les véhicules":                        "Vol",
    "Vols de véhicules":                              "Vol",
    "Vols sans violence contre des personnes":        "Vol",
    "Cambriolages de logement":                       "Cambriolage",
    "Destructions et dégradations volontaires":       "Dégâts",
    "Homicides":                                      "Homicide",
    "Tentatives d'homicide":                          "Homicide",
    "Violences physiques intrafamiliales":            "Violence",
    "Violences physiques hors cadre familial":        "Violence",
    "Coups et blessures volontaires":                 "Violence",
    "Escroqueries et abus de confiance":              "Escroquerie",
    "Escroqueries et fraudes aux moyens de paiement": "Escroquerie",
    "Usage de stupéfiants (AFD)":                     "Stupéfiants",
    "Usage de stupéfiants (hors AFD)":                "Stupéfiants",
    "Trafic de stupéfiants":                          "Stupéfiants",
    "Stupéfiants - infractions à la législation":     "Stupéfiants",
}

_REGION_CODE_RE = re.compile(r"^[0-9A-Za-z]{2}$")


class CrimeRatePredictor:
    """Prédicteur multi-modèles du taux de délinquance régional.

    Entraîne plusieurs algorithmes, sélectionne automatiquement le champion
    sur le critère R² de validation temporelle (TimeSeriesSplit), et expose
    une interface unifiée train / predict / predict_2030 / save / load.

    Parameters
    ----------
    config_path : str | Path, optional
        Chemin vers config.yaml. Si absent, valeurs par défaut utilisées.
    """

    FEATURE_COLS: List[str] = [
        "year_sin",
        "year_cos",
        "year_trend",
        "lag1",
        "lag2",
        "roll_mean_3",
        "region_mean",
        "ind_code",
        "reg_code",
    ]

    _DEFAULT_CONFIG: Dict = {
        "models": {
            "LinearRegression": {
                "class": "LinearRegression",
                "params": {},
            },
            "LightGBM": {
                "class": "LGBMRegressor",
                "params": {
                    "n_estimators": 500,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "num_leaves": 64,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                    "verbose": -1,
                    "n_jobs": 1,
                },
            },
            "RandomForest": {
                "class": "RandomForestRegressor",
                "params": {
                    "n_estimators": 200,
                    "max_depth": 10,
                    "min_samples_split": 5,
                    "random_state": 42,
                    "n_jobs": 1,
                },
            },
            "XGBoost": {
                "class": "XGBRegressor",
                "params": {
                    "n_estimators": 500,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                    "n_jobs": 1,
                },
            },
            "GradientBoosting": {
                "class": "GradientBoostingRegressor",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.05,
                    "max_depth": 5,
                    "random_state": 42,
                },
            },
        },
        "train": {
            "n_cv_splits": 3,
            "test_size": 0.2,
        },
    }

    _MODEL_REGISTRY: Dict = {
        "LinearRegression":          LinearRegression,
        "LGBMRegressor":             LGBMRegressor,
        "RandomForestRegressor":     RandomForestRegressor,
        "XGBRegressor":              XGBRegressor,
        "GradientBoostingRegressor": GradientBoostingRegressor,
    }

    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        self.config = self._load_config(config_path)
        self.model = None
        self.best_model_name: Optional[str] = None
        self.models_metrics: Dict[str, Dict] = {}
        self.feature_names: List[str] = self.FEATURE_COLS

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def _load_config(self, config_path: Optional[Union[str, Path]]) -> Dict:
        """Charge config.yaml ; retourne les defaults si le fichier est absent."""
        if config_path is None:
            logger.warning("Aucun fichier de configuration — valeurs par défaut utilisées.")
            return self._DEFAULT_CONFIG

        path = Path(config_path)
        if not path.exists():
            logger.warning("config.yaml introuvable (%s) — valeurs par défaut utilisées.", path)
            return self._DEFAULT_CONFIG

        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if not cfg or "models" not in cfg:
                logger.warning("config.yaml invalide ou vide — valeurs par défaut utilisées.")
                return self._DEFAULT_CONFIG
            return cfg
        except Exception as exc:
            logger.error("Erreur chargement configuration : %s", exc)
            return self._DEFAULT_CONFIG

    def _build_model(self, model_name: str, params: Dict):
        """Instancie un modèle depuis son nom de classe et ses hyperparamètres."""
        model_cfg = self.config["models"].get(model_name, {})
        class_name = model_cfg.get("class", model_name)
        cls = self._MODEL_REGISTRY.get(class_name)
        if cls is None:
            raise ValueError(
                f"Modèle '{class_name}' non trouvé dans le registre. "
                f"Disponibles : {list(self._MODEL_REGISTRY.keys())}"
            )
        return cls(**params)

    # -------------------------------------------------------------------------
    # Chargement des données
    # -------------------------------------------------------------------------

    def load_data(self, data_path: Union[str, Path]) -> pd.DataFrame:
        """Charge et nettoie les données depuis un fichier local ou une URL.

        Accepte fichier Parquet local, CSV local ou URL distante (sep=';').

        Parameters
        ----------
        data_path : str | Path
            Chemin local ou URL vers le fichier source.

        Returns
        -------
        pd.DataFrame
            DataFrame nettoyé avec taux_100k, Code_region, indicateur, annee.
        """
        data_path = str(data_path)
        logger.info("Chargement des données depuis %s…", data_path[:80])

        try:
            if data_path.endswith(".parquet"):
                df = pd.read_parquet(data_path)
            else:
                df = pd.read_csv(data_path, sep=";", encoding="utf-8", low_memory=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"Fichier introuvable : {data_path}")
        except Exception as exc:
            raise RuntimeError(f"Erreur lecture fichier : {exc}") from exc

        # Vérification des colonnes requises
        required_columns = [
            "annee", "indicateur", "unite_de_compte",
            "nombre", "insee_pop", "CODGEO_2025",
        ]
        missing_columns = [c for c in required_columns if c not in df.columns]
        if missing_columns:
            raise ValueError(f"Colonnes manquantes dans le fichier source : {missing_columns}")

        # Filtre sur le type de compte
        valid_counts = {"Infraction", "nombre"}
        df = df[df["unite_de_compte"].isin(valid_counts)].copy()
        if df.empty:
            raise ValueError(
                f"Aucune ligne avec unite_de_compte dans {valid_counts}. "
                "Vérifier la source de données."
            )

        # Nettoyage numérique
        df["nombre"] = pd.to_numeric(df["nombre"], errors="coerce")
        df["insee_pop"] = pd.to_numeric(df["insee_pop"], errors="coerce")
        df = df[
            df["nombre"].notna()
            & df["insee_pop"].notna()
            & (df["insee_pop"] > 0)
            & (df["nombre"] >= 0)
        ].copy()

        # Calcul du taux pour 100 000 habitants
        df["taux_100k"] = df["nombre"] / df["insee_pop"] * 100_000

        # Extraction du code région depuis CODGEO_2025
        df["Code_region"] = df["CODGEO_2025"].astype(str).str.zfill(2).str[:2]

        # CORRECTION : regex sans antislash devant $ (était r'^[0-9A-Za-z]{2}\$')
        invalid_mask = ~df["Code_region"].str.match(_REGION_CODE_RE.pattern)
        n_invalid = int(invalid_mask.sum())
        if n_invalid > 0:
            logger.warning("%d codes région invalides — remplacés par '00'.", n_invalid)
            df.loc[invalid_mask, "Code_region"] = "00"

        # Mapping indicateurs
        df["indicateur"] = (
            df["indicateur"]
            .astype(str)
            .map(INDICATEUR_MAPPING)
            .fillna(df["indicateur"].astype(str))
        )

        # Forcer str standard (évite Pandas StringDtype)
        df["Code_region"] = df["Code_region"].astype(str)
        df["indicateur"] = df["indicateur"].astype(str)

        df = df.dropna(subset=["taux_100k", "Code_region", "annee"])
        if df.empty:
            raise ValueError("Aucune donnée valide après nettoyage.")

        logger.info("Données chargées : %d lignes.", len(df))
        return df

    # -------------------------------------------------------------------------
    # Feature engineering
    # -------------------------------------------------------------------------

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construit toutes les features à partir du DataFrame brut.

        Accepte un DataFrame issu de load_data() OU un DataFrame déjà
        partiellement préparé (avec taux_100k calculé manuellement).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame avec au minimum : annee, indicateur, CODGEO_2025 (ou
            Code_region), nombre, insee_pop (ou taux_100k déjà calculé).

        Returns
        -------
        pd.DataFrame
            DataFrame avec FEATURE_COLS + annee + indicateur +
            Code_region + taux_100k, sans NaN.
        """
        df = df.copy()

        # Calcul de taux_100k si absent (cas fixture de test)
        if "taux_100k" not in df.columns:
            if "nombre" in df.columns and "insee_pop" in df.columns:
                df["taux_100k"] = df["nombre"] / df["insee_pop"] * 100_000
            else:
                raise ValueError(
                    "Colonne 'taux_100k' absente et impossible à calculer "
                    "(colonnes 'nombre' et 'insee_pop' manquantes)."
                )

        # Extraction de Code_region si absente (cas fixture de test)
        if "Code_region" not in df.columns:
            if "CODGEO_2025" in df.columns:
                df["Code_region"] = df["CODGEO_2025"].astype(str).str.zfill(2).str[:2]
            else:
                raise ValueError(
                    "Colonne 'Code_region' absente et 'CODGEO_2025' manquant."
                )

        # Forcer str standard
        df["Code_region"] = df["Code_region"].astype(str)
        df["indicateur"] = df["indicateur"].astype(str)

        # Tri temporel OBLIGATOIRE avant les lags
        df = df.sort_values(
            ["indicateur", "Code_region", "annee"]
        ).reset_index(drop=True)

        # Features temporelles cycliques
        df["year_sin"] = np.sin(2 * np.pi * df["annee"] / 10)
        df["year_cos"] = np.cos(2 * np.pi * df["annee"] / 10)
        year_min = df["annee"].min()
        year_range = df["annee"].max() - year_min
        df["year_trend"] = (
            (df["annee"] - year_min) / year_range if year_range > 0 else 0.0
        )

        # Lag features — CORRECTION : .bfill() remplace fillna(method='bfill')
        # CORRECTION : transform() évite le MultiIndex produit par
        # groupby().rolling().mean() qui cause un TypeError à l'assignation.
        grp = df.groupby(["indicateur", "Code_region"])["taux_100k"]
        df["lag1"] = grp.shift(1).bfill()
        df["lag2"] = grp.shift(2).bfill()
        df["roll_mean_3"] = grp.transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )

        # Agrégat régional
        df["region_mean"] = df.groupby("Code_region")["taux_100k"].transform("mean")

        # Encodage catégoriel déterministe (pd.Categorical trie alphabétiquement)
        df["ind_code"] = pd.Categorical(df["indicateur"]).codes
        df["reg_code"] = pd.Categorical(df["Code_region"]).codes

        # Colonnes retournées : features + colonnes de contexte utiles aux tests
        output_cols = self.FEATURE_COLS + ["annee", "indicateur", "Code_region", "taux_100k"]
        result = df[output_cols].dropna()

        if result.empty:
            raise ValueError(
                "DataFrame vide après feature engineering. "
                "Vérifier colonnes 'indicateur', 'Code_region', 'annee'."
            )

        logger.info("Features construites : %d observations.", len(result))
        return result

    # -------------------------------------------------------------------------
    # Entraînement multi-modèles
    # -------------------------------------------------------------------------

    def train(
        self,
        data_path: Union[str, Path],
        n_cv_splits: int = 3,
        test_size: float = 0.2,
    ) -> Dict:
        """Entraîne tous les modèles configurés et sélectionne le champion.

        Parameters
        ----------
        data_path : str | Path
            Chemin local ou URL vers le fichier source.
        n_cv_splits : int
            Nombre de folds pour la cross-validation temporelle.
        test_size : float
            Fraction des données réservée au test final (0 < test_size < 1).

        Returns
        -------
        dict
            Métriques du modèle champion avec clé 'r2_score'.
        """
        df = self.load_data(data_path)
        df_feat = self.engineer_features(df)

        X = df_feat[self.FEATURE_COLS]
        y = df_feat["taux_100k"]

        # Split temporel train / test
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Garde une référence au df complet pour predict_2030
        self._last_df_feat = df_feat

        tscv = TimeSeriesSplit(n_splits=n_cv_splits)
        best_cv_r2 = -np.inf
        self.models_metrics = {}

        for model_name, model_cfg in self.config["models"].items():
            params = model_cfg.get("params", {})
            logger.info("Entraînement : %s", model_name)

            cv_scores: List[float] = []
            for train_idx, val_idx in tscv.split(X_train):
                m = self._build_model(model_name, params)
                m.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                cv_scores.append(float(m.score(X_train.iloc[val_idx], y_train.iloc[val_idx])))

            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))

            # Modèle final entraîné sur train complet, évalué sur test
            final_model = self._build_model(model_name, params)
            final_model.fit(X_train, y_train)
            y_pred_test = final_model.predict(X_test)

            # Protection contre test set vide (petits datasets synthétiques)
            if len(y_test) == 0:
                r2_test = float(final_model.score(X_train, y_train))
                y_pred_test = final_model.predict(X_train)
                y_eval = y_train
            else:
                r2_test = float(r2_score(y_test, y_pred_test))
                y_eval = y_test

            self.models_metrics[model_name] = {
                "r2_score":    round(r2_test, 4),
                "rmse":        round(float(np.sqrt(mean_squared_error(y_eval, final_model.predict(X_test if len(y_test) > 0 else X_train)))), 4),
                "mae":         round(float(mean_absolute_error(y_eval, final_model.predict(X_test if len(y_test) > 0 else X_train))), 4),
                "cv_r2_mean":  round(cv_mean, 4),
                "cv_r2_std":   round(cv_std, 4),
            }
            logger.info("  %s → CV R² : %.4f ± %.4f | Test R² : %.4f",
                        model_name, cv_mean, cv_std, r2_test)

            if cv_mean > best_cv_r2:
                best_cv_r2 = cv_mean
                self.model = final_model
                self.best_model_name = model_name

        logger.info("Champion : %s (CV R² = %.4f)", self.best_model_name, best_cv_r2)
        self._log_leaderboard()

        champion = dict(self.models_metrics[self.best_model_name])
        if hasattr(self.model, "feature_importances_"):
            champion["feature_importance"] = dict(
                zip(self.FEATURE_COLS, self.model.feature_importances_.tolist())
            )
        return champion

    def _log_leaderboard(self) -> None:
        """Affiche le tableau comparatif des modèles dans les logs."""
        header = f"{'Modèle':<22} {'R² test':>8} {'RMSE':>9} {'MAE':>9} {'CV R²':>14}"
        logger.info("=" * len(header))
        logger.info(header)
        logger.info("-" * len(header))
        for name, m in sorted(
            self.models_metrics.items(),
            key=lambda x: x[1]["cv_r2_mean"],
            reverse=True,
        ):
            marker = " ← BEST" if name == self.best_model_name else ""
            logger.info(
                "%-22s %8.4f %9.4f %9.4f  %5.4f±%.4f%s",
                name, m["r2_score"], m["rmse"], m["mae"],
                m["cv_r2_mean"], m["cv_r2_std"], marker,
            )
        logger.info("=" * len(header))

    # -------------------------------------------------------------------------
    # Inférence
    # -------------------------------------------------------------------------

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Retourne les prédictions clippées à 0 (taux >= 0 contrainte métier).

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Données d'entrée. Si DataFrame, seules FEATURE_COLS sont utilisées.

        Returns
        -------
        np.ndarray
            Taux prédit pour 100 000 habitants, toujours >= 0.
        """
        if self.model is None:
            raise ValueError("Modèle non chargé. Appelez train() ou load() d'abord.")

        if isinstance(X, pd.DataFrame):
            missing = set(self.FEATURE_COLS) - set(X.columns)
            if missing:
                raise ValueError(f"Colonnes manquantes dans X : {missing}")
            X_input = X[self.FEATURE_COLS]
        else:
            X_input = X

        return np.clip(self.model.predict(X_input), a_min=0, a_max=None)

    def predict_2030(
        self,
        indicateur: str,
        code_region: str,
        df_history: pd.DataFrame,
    ) -> float:
        """Prédit le taux de délinquance pour 2030 pour un indicateur et une région.

        Construit les features 2030 à partir des dernières valeurs historiques
        disponibles dans df_history pour le couple (indicateur, code_region).

        Parameters
        ----------
        indicateur : str
            Catégorie de délinquance (ex: "Vol", "Agression").
        code_region : str
            Code région à 2 caractères (ex: "01", "75").
        df_history : pd.DataFrame
            DataFrame historique (brut ou déjà feature-engineered).

        Returns
        -------
        float
            Taux prédit pour 100 000 habitants en 2030, >= 0.
        """
        if self.model is None:
            raise ValueError("Modèle non chargé. Appelez train() ou load() d'abord.")

        # Préparer les features historiques si nécessaire
        if "lag1" not in df_history.columns:
            df_feat = self.engineer_features(df_history)
        else:
            df_feat = df_history.copy()

        # Filtrer sur l'indicateur et la région demandés
        mask = (
            (df_feat["indicateur"] == indicateur)
            & (df_feat["Code_region"] == code_region)
        )
        subset = df_feat[mask].sort_values("annee")

        # Récupérer les dernières valeurs connues pour construire les lags 2030
        if len(subset) >= 2:
            lag1_val = float(subset["taux_100k"].iloc[-1])
            lag2_val = float(subset["taux_100k"].iloc[-2])
        elif len(subset) == 1:
            lag1_val = float(subset["taux_100k"].iloc[-1])
            lag2_val = lag1_val
        else:
            # Aucune donnée pour ce filtre : utiliser la médiane globale
            lag1_val = float(df_feat["taux_100k"].median())
            lag2_val = lag1_val
            logger.warning(
                "Aucune donnée pour indicateur='%s' region='%s' — médiane utilisée.",
                indicateur, code_region,
            )

        roll_mean = (lag1_val + lag2_val + lag2_val) / 3

        # Encodage catégoriel cohérent avec l'entraînement
        all_indicateurs = sorted(df_feat["indicateur"].unique().tolist())
        all_regions = sorted(df_feat["Code_region"].unique().tolist())
        ind_code = all_indicateurs.index(indicateur) if indicateur in all_indicateurs else 0
        reg_code = all_regions.index(code_region) if code_region in all_regions else 0

        region_mean = float(
            df_feat[df_feat["Code_region"] == code_region]["taux_100k"].mean()
            if (df_feat["Code_region"] == code_region).any()
            else df_feat["taux_100k"].mean()
        )

        TARGET_YEAR = 2030
        year_min = float(df_feat["annee"].min())
        year_max = max(float(df_feat["annee"].max()), TARGET_YEAR)
        year_range = year_max - year_min if year_max > year_min else 1.0

        features_2030 = pd.DataFrame([{
            "year_sin":    np.sin(2 * np.pi * TARGET_YEAR / 10),
            "year_cos":    np.cos(2 * np.pi * TARGET_YEAR / 10),
            "year_trend":  (TARGET_YEAR - year_min) / year_range,
            "lag1":        lag1_val,
            "lag2":        lag2_val,
            "roll_mean_3": roll_mean,
            "region_mean": region_mean,
            "ind_code":    ind_code,
            "reg_code":    reg_code,
        }])

        prediction = float(self.predict(features_2030)[0])
        logger.info(
            "predict_2030 | indicateur=%s region=%s → %.2f (pour 100k hab.)",
            indicateur, code_region, prediction,
        )
        return prediction

    # -------------------------------------------------------------------------
    # Persistance
    # -------------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Sérialise le modèle champion et ses métadonnées.

        Parameters
        ----------
        path : str | Path
            Chemin du fichier de sortie (parents créés si absents).
        """
        if self.model is None:
            raise ValueError("Aucun modèle à sauvegarder. Appelez train() d'abord.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "model":           self.model,
                "feature_names":   self.FEATURE_COLS,
                "best_model_name": self.best_model_name,
                "models_metrics":  self.models_metrics,
                "config":          self.config,
            },
            path,
        )
        logger.info("Modèle sauvegardé : %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CrimeRatePredictor":
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Fichier modèle introuvable : {path}")

        data = joblib.load(path)

        instance = cls()

        if isinstance(data, dict):
            required_keys = {"model", "feature_names", "config"}
            missing_keys = required_keys - set(data.keys())

            if missing_keys:
                raise KeyError(
                    f"Clés manquantes dans le pickle : {missing_keys}. "
                    "Fichier modèle incomplet ou incompatible."
                )

            instance.model = data["model"]
            instance.feature_names = data["feature_names"]
            instance.config = data["config"]
            instance.best_model_name = data.get("best_model_name")
            instance.models_metrics = data.get("models_metrics", {})
        else:
            instance.model = data
            instance.best_model_name = data.__class__.__name__
            instance.models_metrics = {}

            if hasattr(data, "feature_names_in_"):
                instance.feature_names = list(data.feature_names_in_)
            else:
                instance.feature_names = instance.FEATURE_COLS

        logger.info(
            "Modèle '%s' chargé depuis : %s",
            instance.best_model_name,
            path,
        )

        return instance
