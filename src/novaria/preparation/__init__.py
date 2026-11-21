"""Module de préparation PARTAGÉ entre l'entraînement (batch) et le streaming (Flink).

Toute la logique de nettoyage et de transformation vit ici ; elle ne doit
jamais être dupliquée ailleurs (CLAUDE.md, règle 9).

- ``cleaning`` : mise au format, détection et correction des valeurs ;
- ``features`` : variables dérivées (vent, heure, mois en encodage cyclique) ;
- ``quality`` : rapport qualité ;
- ``split`` : découpage chronologique ;
- ``preprocessor`` : imputation et normalisation, intégrées au modèle ;
- ``records`` : préparation de messages unitaires pour le streaming ;
- ``journal`` : journal de vérification des données.
"""

from novaria.preparation.cleaning import (
    clean,
    fix_inconsistencies,
    from_raw,
    standardize,
)
from novaria.preparation.features import add_features
from novaria.preparation.preprocessor import build_preprocessor
from novaria.preparation.records import prepare_record, prepare_records
from novaria.preparation.split import split_chrono

__all__ = [
    "add_features",
    "build_preprocessor",
    "clean",
    "fix_inconsistencies",
    "from_raw",
    "prepare_record",
    "prepare_records",
    "split_chrono",
    "standardize",
]
