"""Détecteurs d'anomalies : les 3 algorithmes avec un seuil calibré de la même façon.

Protocole commun (CLAUDE.md, règle 2) : chaque algorithme produit un score de
normalité (plus il est bas, plus la mesure est atypique). Le seuil d'alerte est
le quantile ``contamination`` des scores sur le train. Ainsi les 3 algorithmes
lèvent la même part d'alertes sur les données d'entraînement et se comparent
à seuil équivalent.

Convention du score (contrat ``/predict``) : ``score = score de normalité -
seuil`` ; une mesure est une anomalie si ``score < 0``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OutlierMixin, clone
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from sklearn.utils.validation import check_is_fitted

from novaria.preparation.preprocessor import build_preprocessor

ALGORITHMS = ("isolation_forest", "local_outlier_factor", "one_class_svm")

# Types déclarés fiables pour le format skops (sérialisation sûre de MLflow,
# qui refuse tout type non listé au chargement). Liste vérifiée pour les 3
# algorithmes ; à passer aussi au chargement du modèle (API, service IA).
SKOPS_TRUSTED_TYPES = [
    "novaria.models.detectors.CalibratedDetector",
    "numpy.dtype",
    "sklearn.tree._tree.Tree",
]


class CalibratedDetector(OutlierMixin, BaseEstimator):
    """Enveloppe un détecteur scikit-learn et calibre son seuil sur le train.

    Attributes:
        estimator: détecteur non ajusté (IsolationForest, LOF en mode
            ``novelty`` ou OneClassSVM).
        contamination: part des données d'entraînement considérées comme
            anomalies, qui fixe le seuil.
        estimator_: détecteur ajusté (après ``fit``).
        threshold_: seuil sur le score de normalité (après ``fit``).
    """

    def __init__(self, estimator: BaseEstimator, contamination: float = 0.01) -> None:
        """Initialise le détecteur, sans l'ajuster.

        Args:
            estimator: détecteur scikit-learn non ajusté.
            contamination: part attendue d'anomalies, entre 0 et 0,5.
        """
        self.estimator = estimator
        self.contamination = contamination

    def fit(self, X: Any, y: Any = None) -> CalibratedDetector:
        """Ajuste le détecteur puis calibre le seuil sur les scores du train.

        Pour LOF, les scores du train sont ceux calculés pendant l'ajustement
        (``negative_outlier_factor_``), car un point du train ne doit pas être
        comparé à lui-même.

        Args:
            X: données d'entraînement (déjà prétraitées).
            y: ignoré, présent pour l'API scikit-learn.

        Returns:
            Le détecteur ajusté.

        Raises:
            ValueError: si ``contamination`` n'est pas dans ]0 ; 0,5].
        """
        if not 0 < self.contamination <= 0.5:
            raise ValueError("contamination doit être dans ]0 ; 0,5]")
        self.estimator_ = clone(self.estimator).fit(X)
        if hasattr(self.estimator_, "negative_outlier_factor_"):
            train_scores = self.estimator_.negative_outlier_factor_
        else:
            train_scores = self.estimator_.score_samples(X)
        self.threshold_ = float(np.quantile(train_scores, self.contamination))
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def score_samples(self, X: Any) -> np.ndarray:
        """Calcule le score de normalité brut de l'algorithme.

        Args:
            X: données prétraitées.

        Returns:
            Un score par ligne ; plus il est bas, plus la ligne est atypique.
        """
        check_is_fitted(self, "threshold_")
        with warnings.catch_warnings():
            # LOF appelle en interne kneighbors sans les noms de colonnes :
            # avertissement scikit-learn sans conséquence.
            warnings.filterwarnings(
                "ignore", message="X does not have valid feature names"
            )
            return self.estimator_.score_samples(X)

    def decision_function(self, X: Any) -> np.ndarray:
        """Calcule le score d'anomalie du contrat : score de normalité moins le seuil.

        Args:
            X: données prétraitées.

        Returns:
            Un score par ligne ; négatif pour une anomalie.
        """
        return self.score_samples(X) - self.threshold_

    def predict(self, X: Any) -> np.ndarray:
        """Prédit anomalie ou non, selon la convention scikit-learn.

        Args:
            X: données prétraitées.

        Returns:
            ``-1`` pour une anomalie, ``1`` pour une mesure normale.
        """
        return np.where(self.decision_function(X) < 0, -1, 1)


def build_estimator(
    algorithm: str, params: dict[str, Any], random_state: int
) -> BaseEstimator:
    """Construit le détecteur scikit-learn d'un algorithme, non ajusté.

    Args:
        algorithm: ``isolation_forest``, ``local_outlier_factor`` ou
            ``one_class_svm``.
        params: hyperparamètres propres à l'algorithme (sans ``scaler``).
        random_state: graine, pour les algorithmes aléatoires.

    Returns:
        Le détecteur non ajusté.

    Raises:
        ValueError: si l'algorithme est inconnu.
    """
    if algorithm == "isolation_forest":
        return IsolationForest(**params, random_state=random_state, n_jobs=-1)
    if algorithm == "local_outlier_factor":
        return LocalOutlierFactor(**params, novelty=True, n_jobs=-1)
    if algorithm == "one_class_svm":
        return OneClassSVM(kernel="rbf", **params)
    raise ValueError(f"Algorithme inconnu : {algorithm!r} (attendu : {ALGORITHMS})")


def build_model(
    algorithm: str, params: dict[str, Any], contamination: float, random_state: int
) -> Pipeline:
    """Construit le pipeline complet non ajusté : préprocesseur puis détecteur calibré.

    C'est ce pipeline qui est enregistré dans MLflow : il reçoit une trame
    préparée (sortie de ``novaria.preparation``) et applique lui-même
    l'imputation et la mise à l'échelle apprises sur le train.

    Args:
        algorithm: nom de l'algorithme (voir ``ALGORITHMS``).
        params: hyperparamètres ; la clé ``scaler`` choisit la mise à l'échelle
            (``standard`` par défaut), les autres vont au détecteur.
        contamination: part attendue d'anomalies, qui fixe le seuil.
        random_state: graine des algorithmes aléatoires.

    Returns:
        Le pipeline scikit-learn non ajusté.
    """
    params = dict(params)
    scaler = params.pop("scaler", "standard")
    detector = CalibratedDetector(
        build_estimator(algorithm, params, random_state), contamination
    )
    return Pipeline(
        [("preprocessor", build_preprocessor(scaler=scaler)), ("detector", detector)]
    )


def anomaly_scores(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Calcule le score d'anomalie du contrat ``/predict`` pour chaque ligne.

    Args:
        model: pipeline ajusté (voir ``build_model``).
        df: trame préparée.

    Returns:
        Un score par ligne ; négatif pour une anomalie.
    """
    return model.decision_function(df)
