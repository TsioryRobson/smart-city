"""Préprocesseur (imputation médiane + standardisation).

Le préprocesseur est ajusté sur le train uniquement, pour éviter toute fuite
d'information vers val et test. Il sera combiné au modèle à l'étape 3 et
n'est donc pas sauvegardé sur disque ici.

Exécution : ``python -m src.preprocess``
"""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURES
from src.data import load_and_split


def build_preprocessor() -> Pipeline:
    """Construit le pipeline imputation médiane -> standardisation.

    Il attend uniquement les colonnes ``FEATURES`` (sans ``station``). Les
    valeurs impossibles, mises à NaN par ``src.data.clean``, sont imputées ici.

    Returns:
        Un ``Pipeline`` scikit-learn non ajusté, qui renvoie des DataFrames.
    """
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return pipeline.set_output(transform="pandas")


def _print_mean_std(name: str, df: pd.DataFrame) -> None:
    """Affiche la moyenne et l'écart-type de chaque variable.

    Sert à contrôler la standardisation : sur le train, on attend une moyenne
    proche de 0 et un écart-type proche de 1 ; sur val et test, des valeurs
    différentes montrent qu'il n'y a pas de fuite.

    Args:
        name: titre affiché avant les valeurs.
        df: DataFrame transformé, avec une colonne par variable.
    """
    print(f"\n{name} : moyenne / écart-type")
    for col in df.columns:
        print(f"  {col:<6} {df[col].mean():>8.3f}  {df[col].std(ddof=0):>8.3f}")


def main() -> None:
    """Ajuste le préprocesseur sur le train et contrôle les transformations."""
    train, val, test = (part[FEATURES] for part in load_and_split())

    preprocessor = build_preprocessor()
    train_t = preprocessor.fit_transform(train)
    val_t = preprocessor.transform(val)
    test_t = preprocessor.transform(test)

    print("\nNaN après transformation (doit être 0) :")
    for name, part in (("train", train_t), ("val", val_t), ("test", test_t)):
        print(f"  {name:<5} {int(part.isna().sum().sum())}")

    _print_mean_std("TRAIN (≈ 0 et ≈ 1)", train_t)
    _print_mean_std("VAL (≠ 0 et ≠ 1 : pas de fuite)", val_t)
    _print_mean_std("TEST (≠ 0 et ≠ 1 : pas de fuite)", test_t)

    imputer: SimpleImputer = preprocessor.named_steps["imputer"]
    print("\nMédianes apprises par l'imputer (train) :")
    for col, value in zip(train.columns, imputer.statistics_, strict=True):
        print(f"  {col:<6} {value:.2f}")


if __name__ == "__main__":
    main()
