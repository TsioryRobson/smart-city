# Note de choix du modèle

Sprint 1 · évaluation sur le test contrôlé (`python -m novaria.pipeline.evaluation`).
Chiffres détaillés : `reports/evaluation.md`, `models/evaluation.json`, expérience MLflow `novaria-air-anomaly` (runs `evaluation`).

## Décision

**Modèle retenu : One-Class SVM** (noyau RBF, `gamma = 0,1`, `nu = 0,01`, `RobustScaler`, seuil calibré à 1 % du train).

| | |
|---|---|
| Run d'entraînement | `a83367e6b7b149f6bac45fe3f07b02b4` |
| Empreinte du modèle (`model_sha256`) | `075aac1834c121fd…` (complète dans `models/selection.json`) |
| Variables | les 17 de `MODEL_FEATURES` |
| Registry (serveur MLflow Docker) | `air-anomaly-detector` v2 = ce modèle (`@champion`) ; v1 = même modèle entraîné sur 2013-09 → 2014-08 (`@previous`) |

## Protocole d'évaluation

- **Mêmes données pour les 3 modèles :** le test (2015-12 → 2016-02, 12 stations, 26 208 lignes), jamais utilisé pour
  l'entraînement ni pour le choix des hyperparamètres. 5 tirages d'anomalies injectées (graines 0 à 4), figés dans
  `data/controlled/` par `novaria.pipeline.injection`, avec empreinte SHA-256 vérifiée avant l'évaluation.
- **Anomalies par tirage :** 524 pics (2 %), 131 valeurs impossibles (0,5 %), 12 séries de capteur figé (≈ 430 lignes).
- **Fausses alertes = alertes sur les heures ordinaires.** Le test contient de vrais épisodes de pollution extrême
  (décembre 2015 : PM2.5 moyen jusqu'à 470 µg/m³ sur une journée). Les heures où un polluant dépasse le 99e centile du
  train (6,7 % des heures de référence) ne sont pas comptées comme fausses alertes : une alerte y est légitime.
  Elles sont mesurées à part (« alertes sur épisodes réels »).
- **Stabilité :** écart-type entre les 5 tirages d'anomalies, et entre 5 réentraînements (mêmes hyperparamètres) sur
  des sous-échantillons de 80 % du train, stratifiés par station et par mois.
- **Règle de sélection** (`configs/evaluation.yaml`) : latence unitaire ≤ 50 ms, puis meilleur Youden des pics
  (détection − fausses alertes) ; à moins de 0,02 d'écart, le moins de fausses alertes l'emporte.
  Même logique que la promotion (détection à 2 points près, fausses alertes inférieures ou égales).

## Résultats

| | Isolation Forest | LOF | **One-Class SVM** |
|---|---|---|---|
| Détection des pics injectés | 70,3 % ± 1,3 | 95,8 % ± 0,8 | **95,9 % ± 0,8** |
| Fausses alertes (heures ordinaires) | 0,04 % | 3,1 % | **2,9 %** |
| Youden des pics | 0,703 | 0,926 | **0,929** |
| AUC des pics | 0,987 | 0,995 | 0,984 |
| Alertes sur épisodes réels extrêmes | 64,1 % | 21,1 % | 52,4 % |
| Détection capteur figé / valeur impossible | 4 % / 3 % | 12 % / 10 % | 10 % / 10 % |
| Youden : écart-type entre tirages | 0,013 | 0,008 | 0,009 |
| Youden : écart-type entre réentraînements | 0,016 | 0,002 | **0,001** |
| Latence unitaire | 11 ms | 15 ms | **8 ms** |
| Taille du modèle | 9 Mo | 100 Mo | **0,4 Mo** |

## Justification

1. **Meilleure détection à fausses alertes égales.** One-Class SVM et LOF détectent 96 % des pics injectés ; leurs
   Youden (0,929 et 0,926) sont à moins de 0,02 l'un de l'autre. One-Class SVM a moins de fausses alertes (2,9 % contre
   3,1 %) : il est retenu par la règle.
2. **Le plus stable.** Réentraîné sur 5 sous-échantillons différents, son Youden ne varie que de 0,922 à 0,924
   (écart-type 0,001) : le résultat ne dépend pas du hasard de l'échantillon. Point important pour le réentraînement
   automatique du sprint 3.
3. **Le plus léger et le plus rapide.** 0,4 Mo et 8 ms par mesure, contre 100 Mo pour LOF (qui stocke tout le train).
   Adapté à l'API, au rechargement à chaud et à un futur déploiement sur Kubernetes.
4. **Meilleur que LOF sur les vrais épisodes.** Il alerte sur 52 % des heures de pollution réellement extrême, contre
   21 % pour LOF : LOF juge « normal » un épisode s'il ressemble à ses voisins du train.

## Alternative écartée : Isolation Forest

Isolation Forest est le choix inverse : **quasiment aucune fausse alerte** (0,04 %, soit environ 0,1 par jour sur
12 stations) et la meilleure couverture des vrais épisodes (64 %), mais il **manque 30 % des pics injectés** à ce seuil.
Son AUC élevée (0,987) montre que ses scores sont bien ordonnés : un seuil plus souple en rattraperait une partie.

L'équipe a retenu la règle centrée sur la détection : One-Class SVM. Isolation Forest reste l'alternative documentée
si la démonstration montre trop de fausses alertes. Il est aussi le plus rapide à réentraîner (1 s).

## Points faibles connus de One-Class SVM

- **≈ 8,5 fausses alertes par jour** sur 12 stations (2,9 % des heures ordinaires), concentrées en janvier 2016
  (7,8 %, contre 0,7 % en décembre et 0,3 % en février).
- **70 % de ces fausses alertes tombent sur une météo jamais vue au train** : la vague de froid record des
  22-24 janvier 2016 (-19,9 °C, contre -13 °C au minimum dans le train), avec un air pourtant propre. Le modèle signale
  une situation atypique, mais pas une pollution atypique.
- Pistes : la règle métier de l'API (`alert_type`) peut ne lever `pollution_atypique` que si un polluant est en cause ;
  la fenêtre glissante de 12 mois intégrera ces conditions au réentraînement ; le moniteur du sprint 3 surveille le
  taux de fausses alertes (déclencheur à 30 % sur 7 jours).
- **Capteur figé et valeurs impossibles ne sont pas détectés par le modèle** (≈ 10 %) : un modèle point par point ne
  voit ni une répétition, ni une valeur déjà mise à NaN par le nettoyage. Ils relèvent de règles (`defaut_capteur`).

## Limites de l'évaluation

- **Anomalies synthétiques :** les pics sont nets (au-dessus du 99e centile). Les taux de détection ne valident pas
  une performance réelle (règle 7).
- **Test hivernal uniquement :** les fausses alertes dépendent de la saison (validation d'automne : 1,4 %).
- **Épisodes réels approximés** par le seuil du 99e centile, une définition proche de celle des pics injectés :
  elle favorise les modèles qui réagissent aux valeurs extrêmes d'une seule variable (Isolation Forest).
- À la validation, les fausses alertes incluaient encore les épisodes réels : les chiffres de validation et de test
  ne sont donc pas directement comparables sur ce point.

## Model Registry

Deux versions de One-Class SVM (mêmes hyperparamètres) qui ne diffèrent que par la fenêtre de données, comparées
par `promotion.py` sur le même test contrôlé (2015-12 → 2016-02, postérieur aux deux fenêtres) :

| Version | Fenêtre d'entraînement | Détection des pics | Fausses alertes | Latence | Décision | Alias |
|---|---|---|---|---|---|---|
| v1 | 2013-09 → 2014-08 | 94,1 % | 3,57 % | 12,8 ms | promue (aucun champion) | `@previous` |
| v2 | 2014-09 → 2015-08 | 95,9 % | 2,94 % | 7,3 ms | **promue par la règle** (détection ≥ v1 − 2 points, fausses alertes ≤ v1) | `@champion` |

La promotion n'a pas eu besoin d'être forcée : la fenêtre la plus récente est meilleure sur les deux critères, ce qui
justifie la fenêtre glissante de 12 mois (règle 8).
