# Préparation des données : décisions et justifications

Sprint 1 · module `novaria.preparation`, scripts `novaria.pipeline.ingestion` et `novaria.pipeline.preparation`.
Les chiffres viennent du notebook `notebooks/01_exploration.ipynb`, sur les 12 stations (420 768 lignes).

## 1. Périmètre des données

| Décision | Justification |
|---|---|
| Entraînement sur les 12 stations | Plus de diversité (urbain, périurbain) et 12 fois plus de données qu'une station. |
| Station de démo : Aotizhongxin | Station du contrat d'API ; ses données 2016-03 → 2017-02 sont rejouées par le simulateur. |
| Train 2014-09 → 2015-08, val 2015-09 → 2015-11, test 2015-12 → 2016-02 | Fenêtre de 12 mois pour couvrir toutes les saisons, puis 3 + 3 mois. Découpage par dates, jamais aléatoire. |
| 2016-03 → 2017-02 exclu | Période de démo : le modèle ne doit jamais l'avoir vue. Une assertion le vérifie. |
| 2013-03 → 2014-08 non utilisé | Réservé à une deuxième version du modèle (autre fenêtre de données) ou au réentraînement. |

## 2. Variables

| Décision | Justification |
|---|---|
| 12 mesures du contrat `/predict` | Le modèle doit recevoir ce que le capteur envoie : même schéma en batch et en streaming. |
| Vent `wd` en sin/cos | 16 directions circulaires : NNW est proche de N. Un encodage one-hot perdrait cette proximité. |
| Heure et mois en sin/cos | La pollution dépend fortement de la saison et de l'heure (voir §5). Sans ce contexte, une pollution hivernale normale passerait pour une anomalie. |
| Horodatages en heure locale de Pékin | Le dataset est en heure locale. Un message avec fuseau est converti, pour que l'heure encodée soit la même qu'à l'entraînement. |

## 3. Nettoyage

| Problème | Constat | Décision |
|---|---|---|
| Doublons (station, horodatage) | 0 | Supprimés s'il y en a (première occurrence gardée). Seule suppression de lignes autorisée. |
| Heures manquantes | 0 (grille horaire complète) | Signalées ; contrôle utile en streaming. |
| Valeurs manquantes | 0,1 % (météo) à 4,9 % (CO), jusqu'à 9,1 % (CO à Dongsi) | Signalées, imputées par la médiane du train dans le pipeline du modèle. |
| Valeurs négatives | 0 polluant négatif ; températures et points de rosée négatifs réels | Mises à NaN pour les polluants, la pluie et le vent uniquement. |
| PM2.5 > PM10 | 21 312 lignes (5,1 %) ; écart médian 19 µg/m³ ; 9,5 % sous 5 % d'écart | Physiquement impossible (PM2.5 ⊂ PM10) et pas du bruit de mesure : **les deux valeurs** sont mises à NaN. Décision validée par l'équipe malgré le volume. |
| O3 > 800 µg/m³ | 16 valeurs, toutes à Dongsi en août 2013, presque toutes égales à 1 071 | Valeur bloquée d'un capteur défaillant (même valeur pendant 16 h, sans lien avec NO2). Borne haute O3 fixée à 800 µg/m³, plus de 3 fois le p99 du dataset (245) : choix de l'équipe, à revoir si d'autres capteurs dépassent ce seuil. |
| Plafonds de capteur | CO = 10 000 (56 lignes), PM = 999 (4 lignes) | Gardés : extrêmes mais possibles, ce sont les anomalies à détecter. |
| Capteur figé | 1 267 séries ≥ 24 h, dont 1 051 en SO2, surtout à 2 µg/m³ | Signalé seulement. Seuil de 24 h car SO2 et CO sont mesurés par paliers. La pluie est exclue (0 la plupart du temps). |
| Direction de vent inconnue | 0 | Mise à NaN (règle utile pour les messages du simulateur). |

### Bornes physiques

| Variable | Bornes | Variable | Bornes |
|---|---|---|---|
| pm25, pm10 | 0 – 1 000 µg/m³ | temp | -40 – 50 °C |
| no2, so2 | 0 – 1 000 µg/m³ | pres | 870 – 1 085 hPa (records mondiaux) |
| co | 0 – 20 000 µg/m³ | dewp | -60 – 35 °C |
| o3 | 0 – 800 µg/m³ | rain | 0 – 200 mm/h |
| | | wspm | 0 – 75 m/s |

Les bornes rejettent uniquement l'impossible. Toutes les valeurs du dataset y entrent, sauf les 16 valeurs d'O3.

## 4. Imputation et normalisation dans le modèle

Le préprocesseur (`SimpleImputer(median)` + `StandardScaler`) n'est **pas** appliqué par la préparation :
`training.py` le met dans le même pipeline scikit-learn que le modèle, enregistré dans MLflow.

- **Pas de fuite :** il est ajusté sur le train uniquement.
- **Pas d'écart batch / streaming :** l'API et le service IA appliquent le même artefact ; Flink ne fait que le nettoyage
  et les variables dérivées, avec les mêmes fonctions que le batch (`prepare_records`, testé contre le batch).
- **Limite :** une valeur imputée paraît « normale ». Les queues de distribution très longues (asymétrie 2 à 3)
  rendent le `StandardScaler` sensible : à comparer avec `RobustScaler` à l'entraînement.

## 5. Saisonnalité et limites de l'évaluation

| Variable | Hiver | Printemps | Été | Automne |
|---|---|---|---|---|
| SO2 (médiane, µg/m³) | 19 | 11 | 3 | 5 |
| O3 | 21 | 67 | 78 | 21 |
| PM10 | 77 | 104 | 70 | 80 |
| CO | 1 200 | 800 | 800 | 1 000 |

Chauffage au charbon l'hiver (SO2, CO), photochimie l'été (O3), poussières au printemps (PM10).
Le test (2015-12 → 2016-02) est **uniquement hivernal** : température médiane -1,3 °C contre 14,6 °C au train.
Le taux de fausses alertes mesuré sur le test ne vaut donc que pour l'hiver.

## 6. Traçabilité

- `data/ingested/manifest.json` : empreinte SHA-256 de chaque CSV et du jeu (`dataset_sha256`), à enregistrer dans MLflow.
- `data/prepared/metadata.json` : bornes du split, variables, empreinte de chaque jeu préparé.
- `logs/data_checks.log` : une ligne par problème détecté et par action appliquée.
