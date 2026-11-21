# Évaluation des modèles sur le test contrôlé

Généré le 2026-11-21T17:36:20+00:00 · commit `e8f0636836773cc5ae229321a43e304e9de306a1-dirty` · test 2015-12 → 2016-02 · 5 tirages d'anomalies · 5 réentraînements à 80% du train

## Performance (moyenne ± écart-type sur les tirages d'anomalies)

Fausses alertes : alertes sur les heures de référence **ordinaires**. Les heures où un polluant dépasse le 99e centile du train sont de vrais épisodes de pollution extrême (non étiquetés) : une alerte y est légitime, elle est comptée à part.

| Modèle | Détection pics | Fausses alertes | Youden pics | AUC pics | Détection figé | Détection impossible |
|---|---|---|---|---|---|---|
| isolation_forest | 70.3 % ± 1.3 | 0.0 % ± 0.0 | 0.703 ± 0.013 | 0.987 ± 0.003 | 4.4 % ± 2.6 | 3.4 % ± 1.0 |
| local_outlier_factor | 95.8 % ± 0.8 | 3.1 % ± 0.1 | 0.926 ± 0.008 | 0.995 ± 0.001 | 11.8 % ± 4.9 | 10.1 % ± 1.9 |
| one_class_svm | 95.9 % ± 0.8 | 2.9 % ± 0.1 | 0.929 ± 0.009 | 0.984 ± 0.004 | 10.4 % ± 5.1 | 9.6 % ± 1.8 |

## Alertes sur les données réelles de référence

Épisodes réels de pollution extrême : 6.7 % des heures de référence du test.

| Modèle | Alertes sur toute la référence | Alertes sur épisodes réels | Fausses alertes sur heures ordinaires | Part des fausses alertes avec météo hors plage du train |
|---|---|---|---|---|
| isolation_forest | 4.4 % | 64.1 % | 0.0 % | 45.5 % |
| local_outlier_factor | 4.3 % | 21.1 % | 3.1 % | 58.2 % |
| one_class_svm | 6.3 % | 52.4 % | 2.9 % | 70.2 % |

## Stabilité

| Modèle | Youden : écart-type entre tirages | Youden : écart-type entre réentraînements | Youden : min – max réentraînements | Fausses alertes par mois | Fausses alertes : station max |
|---|---|---|---|---|---|
| isolation_forest | 0.0129 | 0.0157 | 0.671 – 0.714 | 2015-12 0.1 %, 2016-01 0.1 %, 2016-02 0.0 % | Shunyi 0.3 % |
| local_outlier_factor | 0.0079 | 0.0022 | 0.912 – 0.917 | 2015-12 1.8 %, 2016-01 6.4 %, 2016-02 0.8 % | Dingling 7.4 % |
| one_class_svm | 0.0085 | 0.0010 | 0.922 – 0.924 | 2015-12 0.7 %, 2016-01 7.8 %, 2016-02 0.3 % | Shunyi 4.2 % |

## Coût

| Modèle | Latence unitaire (ms) | Inférence (ms / ligne) | Taille du modèle |
|---|---|---|---|
| isolation_forest | 10.7 | 0.005 | 9.3 Mo |
| local_outlier_factor | 20.8 | 0.202 | 100.3 Mo |
| one_class_svm | 5.0 | 0.066 | 0.4 Mo |

## Sélection

**Modèle retenu : `one_class_svm`**

- one_class_svm : test_youden_pic moyen 0.929 (retenu dans le groupe de tête).
- local_outlier_factor : test_youden_pic moyen 0.926 (retenu dans le groupe de tête).
- isolation_forest : test_youden_pic moyen 0.703 (écarté).
- Écart inférieur à 0.02 entre local_outlier_factor, one_class_svm : one_class_svm a le moins de fausses alertes.

Les anomalies sont synthétiques : ces chiffres ne valident pas une performance réelle (CLAUDE.md, règle 7). Le test ne couvre que l'hiver.
