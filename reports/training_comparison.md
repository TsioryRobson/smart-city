# Comparaison des algorithmes (validation contrôlée)

Généré le 2026-11-21T17:05:41+00:00 · commit `3fbd2d66cf5bb74b7550ad38e35b5dab81d38c25-dirty` · jeu `439a2ed197c6` · contamination 1%

Validation : 26208 lignes, dont 524 `pic`, 131 `valeur_impossible`, 451 `capteur_fige`. Critère de choix du réglage : `val_youden_pic`.

## Modèles retenus (meilleur réglage par algorithme)

| Modèle | Détection pics | Fausses alertes | Youden pics | AUC pics | Détection figé | Détection impossible | Entraînement (s) | Inférence (ms/ligne) | Latence unitaire (ms) |
|---|---|---|---|---|---|---|---|---|---|
| isolation_forest | 63.9% | 1.15% | 0.628 | 0.978 | 0.2% | 1.5% | 1.3 | 0.005 | 11.5 |
| local_outlier_factor | 98.3% | 3.35% | 0.949 | 0.996 | 3.3% | 9.2% | 20.5 | 0.195 | 10.7 |
| one_class_svm | 96.4% | 1.44% | 0.949 | 0.994 | 0.7% | 3.1% | 52.2 | 0.056 | 5.8 |

## Tous les réglages testés

| Réglage | Détection pics | Fausses alertes | Youden pics | AUC pics | Détection figé | Détection impossible | Entraînement (s) | Inférence (ms/ligne) | Latence unitaire (ms) |
|---|---|---|---|---|---|---|---|---|---|
| isolation_forest (max_features=1.0, max_samples=256, n_estimators=100, scaler=standard) | 40.6% | 0.91% | 0.397 | 0.957 | 0.0% | 0.0% | 0.7 | 0.003 | 10.1 |
| isolation_forest (max_features=1.0, max_samples=256, n_estimators=300, scaler=standard) | 36.6% | 0.84% | 0.358 | 0.964 | 0.0% | 0.0% | 1.7 | 0.012 | 24.9 |
| isolation_forest (max_features=1.0, max_samples=2048, n_estimators=100, scaler=standard) | 63.9% | 1.15% | 0.628 | 0.978 | 0.2% | 1.5% | 1.3 | 0.005 | 11.5 |
| isolation_forest (max_features=1.0, max_samples=2048, n_estimators=300, scaler=standard) | 63.2% | 1.06% | 0.621 | 0.979 | 0.0% | 0.8% | 3.2 | 0.024 | 33.7 |
| isolation_forest (max_features=0.7, max_samples=256, n_estimators=100, scaler=standard) | 29.8% | 0.75% | 0.290 | 0.958 | 0.0% | 0.0% | 1.1 | 0.005 | 14.5 |
| isolation_forest (max_features=0.7, max_samples=256, n_estimators=300, scaler=standard) | 33.8% | 0.78% | 0.330 | 0.957 | 0.0% | 0.8% | 2.6 | 0.011 | 27.2 |
| isolation_forest (max_features=0.7, max_samples=2048, n_estimators=100, scaler=standard) | 59.9% | 0.90% | 0.590 | 0.978 | 0.0% | 0.8% | 2.0 | 0.010 | 17.8 |
| isolation_forest (max_features=0.7, max_samples=2048, n_estimators=300, scaler=standard) | 60.7% | 0.94% | 0.598 | 0.979 | 0.0% | 0.0% | 3.8 | 0.017 | 34.1 |
| local_outlier_factor (n_neighbors=20, scaler=standard) | 98.7% | 10.55% | 0.881 | 0.994 | 14.4% | 22.9% | 22.8 | 0.275 | 17.7 |
| local_outlier_factor (n_neighbors=20, scaler=robust) | 97.3% | 8.49% | 0.888 | 0.990 | 14.2% | 19.8% | 28.6 | 0.254 | 15.4 |
| local_outlier_factor (n_neighbors=50, scaler=standard) | 98.9% | 5.96% | 0.929 | 0.997 | 8.2% | 14.5% | 24.8 | 0.235 | 17.8 |
| local_outlier_factor (n_neighbors=50, scaler=robust) | 96.2% | 5.12% | 0.911 | 0.992 | 9.3% | 11.5% | 23.1 | 0.201 | 12.7 |
| local_outlier_factor (n_neighbors=100, scaler=standard) | 98.3% | 3.35% | 0.949 | 0.996 | 3.3% | 9.2% | 20.5 | 0.195 | 10.7 |
| local_outlier_factor (n_neighbors=100, scaler=robust) | 94.3% | 3.26% | 0.910 | 0.990 | 4.9% | 8.4% | 23.7 | 0.262 | 34.6 |
| one_class_svm (gamma=scale, nu=0.01, scaler=standard) | 96.6% | 2.04% | 0.945 | 0.993 | 1.3% | 3.1% | 15.9 | 0.069 | 9.5 |
| one_class_svm (gamma=scale, nu=0.01, scaler=robust) | 96.2% | 1.40% | 0.948 | 0.994 | 0.4% | 3.1% | 15.6 | 0.054 | 6.2 |
| one_class_svm (gamma=0.02, nu=0.01, scaler=standard) | 82.3% | 0.83% | 0.814 | 0.967 | 0.0% | 0.8% | 15.1 | 0.054 | 5.1 |
| one_class_svm (gamma=0.02, nu=0.01, scaler=robust) | 80.3% | 0.69% | 0.797 | 0.954 | 0.0% | 0.0% | 16.0 | 0.056 | 6.2 |
| one_class_svm (gamma=0.1, nu=0.01, scaler=standard) | 98.5% | 3.98% | 0.945 | 0.997 | 4.2% | 3.8% | 55.3 | 0.061 | 5.7 |
| one_class_svm (gamma=0.1, nu=0.01, scaler=robust) | 96.4% | 1.44% | 0.949 | 0.994 | 0.7% | 3.1% | 52.2 | 0.056 | 5.8 |

Le seuil de chaque modèle est calibré pour 1 alerte sur 100 mesures du train. Le capteur figé et les valeurs impossibles relèvent surtout des règles (nettoyage, contrôle « capteur figé ») : leur détection par le modèle est indiquée pour information. Résultats sur la validation seulement : le test contrôlé est réservé à `evaluation.py`.
