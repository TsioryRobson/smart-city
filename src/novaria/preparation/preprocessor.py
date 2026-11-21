"""Préprocesseur du modèle : sélection des variables, imputation médiane, standardisation.

Il n'est pas sauvegardé par la préparation : ``training.py`` le met dans le
même pipeline scikit-learn que le modèle, enregistré dans MLflow. Ainsi
l'imputation et la normalisation sont ajustées sur le train uniquement (pas de
fuite) et appliquées à l'identique en batch et en streaming. Flink ne fait que
le nettoyage et les variables dérivées.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from novaria.common.config import MODEL_FEATURES

SCALERS = {"standard": StandardScaler, "robust": RobustScaler}


def build_preprocessor(
    features: list[str] | None = None, scaler: str = "standard"
) -> ColumnTransformer:
    """Construit le préprocesseur non ajusté : sélection, imputation médiane, mise à l'échelle.

    Il accepte une trame préparée complète : seules les colonnes ``features``
    sont gardées (station, direction brute du vent, etc. sont ignorées). Les
    valeurs impossibles, mises à NaN par le nettoyage, sont imputées ici.

    Args:
        features: colonnes d'entrée du modèle (défaut : ``MODEL_FEATURES``).
        scaler: ``standard`` (moyenne / écart-type) ou ``robust`` (médiane /
            écart interquartile, moins sensible aux queues de distribution).

    Returns:
        Un ``ColumnTransformer`` scikit-learn non ajusté, qui renvoie des
        DataFrames avec les noms de colonnes d'origine.

    Raises:
        ValueError: si ``scaler`` n'est pas une valeur connue.
    """
    features = MODEL_FEATURES if features is None else features
    if scaler not in SCALERS:
        raise ValueError(f"scaler inconnu : {scaler!r} (attendu : {sorted(SCALERS)})")
    numeric = Pipeline(
        [
            # keep_empty_features : une variable vide au train reste en sortie
            # (imputée à 0), pour que le modèle reçoive toujours MODEL_FEATURES.
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", SCALERS[scaler]()),
        ]
    )
    preprocessor = ColumnTransformer(
        [("features", numeric, features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor.set_output(transform="pandas")
