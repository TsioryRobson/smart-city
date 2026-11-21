# Comparaison des algorithmes (validation contrôlée)

Généré le 2026-11-21T19:26:29+00:00 · commit `dca408ff15ff7df7c3985baf1130d994fa8e2ab9-dirty` · jeu `439a2ed197c6` · contamination 1%

Validation : 26208 lignes, dont 524 `pic`, 131 `valeur_impossible`, 451 `capteur_fige`. Critère de choix du réglage : `val_youden_pic`.

## Modèles retenus (meilleur réglage par algorithme)

| Modèle | Détection pics | Fausses alertes | Youden pics | AUC pics | Détection figé | Détection impossible | Entraînement (s) | Inférence (ms/ligne) | Latence unitaire (ms) |
|---|---|---|---|---|---|---|---|---|---|
| isolation_forest | 63.9% | 1.15% | 0.628 | 0.978 | 0.2% | 1.5% | 1.2 | 0.006 | 12.6 |
| local_outlier_factor | 98.3% | 3.35% | 0.949 | 0.996 | 3.3% | 9.2% | 23.6 | 0.202 | 18.5 |
| one_class_svm | 96.4% | 1.44% | 0.949 | 0.994 | 0.7% | 3.1% | 45.5 | 0.053 | 5.6 |

## Tous les réglages testés

| Réglage | Détection pics | Fausses alertes | Youden pics | AUC pics | Détection figé | Détection impossible | Entraînement (s) | Inférence (ms/ligne) | Latence unitaire (ms) |
|---|---|---|---|---|---|---|---|---|---|
| isolation_forest (max_features=1.0, max_samples=256, n_estimators=100, scaler=standard) | 40.6% | 0.91% | 0.397 | 0.957 | 0.0% | 0.0% | 1.0 | 0.004 | 12.7 |
| isolation_forest (max_features=1.0, max_samples=256, n_estimators=300, scaler=standard) | 36.6% | 0.84% | 0.358 | 0.964 | 0.0% | 0.0% | 2.3 | 0.011 | 28.3 |
| isolation_forest (max_features=1.0, max_samples=2048, n_estimators=100, scaler=standard) | 63.9% | 1.15% | 0.628 | 0.978 | 0.2% | 1.5% | 1.2 | 0.006 | 12.6 |
| isolation_forest (max_features=1.0, max_samples=2048, n_estimators=300, scaler=standard) | 63.2% | 1.06% | 0.621 | 0.979 | 0.0% | 0.8% | 3.4 | 0.016 | 27.5 |
| isolation_forest (max_features=0.7, max_samples=256, n_estimators=100, scaler=standard) | 29.8% | 0.75% | 0.290 | 0.958 | 0.0% | 0.0% | 1.0 | 0.004 | 13.0 |
| isolation_forest (max_features=0.7, max_samples=256, n_estimators=300, scaler=standard) | 33.8% | 0.78% | 0.330 | 0.957 | 0.0% | 0.8% | 2.5 | 0.013 | 35.5 |
| isolation_forest (max_features=0.7, max_samples=2048, n_estimators=100, scaler=standard) | 59.9% | 0.90% | 0.590 | 0.978 | 0.0% | 0.8% | 1.2 | 0.006 | 12.8 |
| isolation_forest (max_features=0.7, max_samples=2048, n_estimators=300, scaler=standard) | 60.7% | 0.94% | 0.598 | 0.979 | 0.0% | 0.0% | 3.5 | 0.018 | 27.5 |
| local_outlier_factor (n_neighbors=20, scaler=standard) | 98.7% | 10.55% | 0.881 | 0.994 | 14.4% | 22.9% | 18.6 | 0.173 | 18.3 |
| local_outlier_factor (n_neighbors=20, scaler=robust) | 97.3% | 8.49% | 0.888 | 0.990 | 14.2% | 19.8% | 19.1 | 0.191 | 20.3 |
| local_outlier_factor (n_neighbors=50, scaler=standard) | 98.9% | 5.96% | 0.929 | 0.997 | 8.2% | 14.5% | 23.4 | 0.215 | 19.1 |
| local_outlier_factor (n_neighbors=50, scaler=robust) | 96.2% | 5.12% | 0.911 | 0.992 | 9.3% | 11.5% | 20.1 | 0.177 | 15.9 |
| local_outlier_factor (n_neighbors=100, scaler=standard) | 98.3% | 3.35% | 0.949 | 0.996 | 3.3% | 9.2% | 23.6 | 0.202 | 18.5 |
| local_outlier_factor (n_neighbors=100, scaler=robust) | 94.3% | 3.26% | 0.910 | 0.990 | 4.9% | 8.4% | 22.2 | 0.199 | 17.6 |
| one_class_svm (gamma=scale, nu=0.01, scaler=standard) | 96.6% | 2.04% | 0.945 | 0.993 | 1.3% | 3.1% | 14.9 | 0.049 | 5.0 |
| one_class_svm (gamma=scale, nu=0.01, scaler=robust) | 96.2% | 1.40% | 0.948 | 0.994 | 0.4% | 3.1% | 16.2 | 0.054 | 6.9 |
| one_class_svm (gamma=0.02, nu=0.01, scaler=standard) | 82.3% | 0.83% | 0.814 | 0.967 | 0.0% | 0.8% | 17.6 | 0.050 | 5.9 |
| one_class_svm (gamma=0.02, nu=0.01, scaler=robust) | 80.3% | 0.69% | 0.797 | 0.954 | 0.0% | 0.0% | 15.1 | 0.051 | 5.8 |
| one_class_svm (gamma=0.1, nu=0.01, scaler=standard) | 98.5% | 3.98% | 0.945 | 0.997 | 4.2% | 3.8% | 55.5 | 0.058 | 5.5 |
| one_class_svm (gamma=0.1, nu=0.01, scaler=robust) | 96.4% | 1.44% | 0.949 | 0.994 | 0.7% | 3.1% | 45.5 | 0.053 | 5.6 |

Le seuil de chaque modèle est calibré pour 1 alerte sur 100 mesures du train. Le capteur figé et les valeurs impossibles relèvent surtout des règles (nettoyage, contrôle « capteur figé ») : leur détection par le modèle est indiquée pour information. Résultats sur la validation seulement : le test contrôlé est réservé à `evaluation.py`.
