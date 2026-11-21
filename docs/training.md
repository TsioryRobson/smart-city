# Entraînement des algorithmes : protocole, réglages et résultats

Sprint 1 · `python -m novaria.pipeline.training` · configuration `configs/training.yaml`.
Résultats complets : `reports/training_comparison.md` et l'expérience MLflow `novaria-air-anomaly`.

## 1. Protocole commun aux 3 algorithmes

| Décision | Justification |
|---|---|
| Un pipeline par modèle : préprocesseur (imputation médiane + mise à l'échelle) puis détecteur | Le préprocesseur est ajusté sur le train uniquement (pas de fuite) et voyage avec le modèle dans MLflow : batch et streaming appliquent exactement la même transformation. |
| Mêmes 17 variables pour tous (`MODEL_FEATURES`) | Comparaison à entrée égale. |
| Seuil = quantile 1 % des scores du train (`CalibratedDetector`) | Les 3 algorithmes n'ont pas le même réglage natif du seuil (`contamination`, `nu`). On calibre le seuil de la même façon pour tous : chacun alerte sur 1 % du train, donc on les compare à seuil équivalent. |
| Contamination 1 % | Environ 1 alerte pour 100 mesures, soit 1 par station tous les 4 jours. Sert aussi de référence au déclencheur « taux d'alerte » du sprint 3. |
| Score = score de normalité − seuil ; anomalie si score < 0 | Convention du contrat `/predict` (`score: -0.21`, `is_anomaly: true`). |
| Graine fixe (`random_state: 42`) | Résultats reproductibles (Isolation Forest, injection). |

## 2. Validation contrôlée

Les anomalies réelles ne sont pas étiquetées (règle 3). Pour choisir les hyperparamètres, on injecte des anomalies connues
dans la **validation** (2015-09 → 2015-11) avec `novaria.common.injection` ; le train n'en reçoit jamais (règle 4).

| Type | Volume | Construction | Rôle attendu |
|---|---|---|---|
| `pic` | 2 % des lignes (524) | 1 à 3 polluants portés au-dessus du 99e centile (×1 à ×1,5) ou à 3-6 fois leur valeur, sans dépasser la borne physique ni créer PM2.5 > PM10 | **Modèle** : valeur possible mais atypique |
| `valeur_impossible` | 0,5 % (131) | Négatif, 1,5 × la borne haute, ou PM2.5 > PM10 | **Nettoyage** : mise à NaN puis imputée |
| `capteur_fige` | 12 séries de 24 à 48 h (451 lignes) | Une mesure répète la valeur de l'heure précédente | **Règle « capteur figé »** (historique par station) |

Après injection, le même nettoyage qu'en production est réappliqué (`fix_inconsistencies`) : une valeur impossible
arrive au modèle sous forme de NaN imputé, comme en streaming. Le **test contrôlé** (2015-12 → 2016-02) n'est pas
utilisé ici : il est réservé à `evaluation.py`, pour une mesure non biaisée par le choix des réglages.

## 3. Critère de choix du réglage

**Indice de Youden des pics = taux de détection des pics − taux de fausses alertes** (sur la validation), à maximiser ;
à égalité, le moins de fausses alertes.

- Seuls les pics comptent : ce sont les seules anomalies dont la détection relève du modèle. Le capteur figé et les
  valeurs impossibles sont traités par des règles ; leur détection par le modèle est mesurée pour information.
- La détection et les fausses alertes sont les deux critères de la promotion (CLAUDE.md, section 7).
- L'AUC ROC et la précision moyenne des pics (indépendantes du seuil) sont aussi enregistrées.

## 4. Grilles d'hyperparamètres

| Algorithme | Grille | Justification |
|---|---|---|
| Isolation Forest | `n_estimators` ∈ {100, 300} × `max_samples` ∈ {256, 2048} × `max_features` ∈ {1,0 ; 0,7} | Nombre d'arbres (stabilité), taille d'échantillon par arbre (256 = valeur de l'article d'origine ; 2048 pour mieux couvrir 12 stations × 4 saisons), part des variables. Mise à l'échelle sans effet sur des arbres : non comparée. |
| LOF (`novelty=True`) | `n_neighbors` ∈ {20, 50, 100} × scaler ∈ {standard, robust} | Taille du voisinage (petite = sensible au bruit, grande = lisse les régimes saisonniers). |
| One-Class SVM (RBF) | `gamma` ∈ {scale, 0,02 ; 0,1} × scaler ∈ {standard, robust}, `nu` = 0,01 | Largeur du noyau. `nu` aligné sur la contamination ; le seuil final est de toute façon recalibré. |

Le `RobustScaler` (médiane, écart interquartile) est comparé au `StandardScaler` car l'exploration a montré des queues
de distribution très longues (asymétrie 2 à 3 pour PM, CO, SO2).

### Échantillonnage

Mesuré sur les 105 120 lignes du train : LOF 19 à 38 s, One-Class SVM 18 à 62 s, Isolation Forest 1 à 4 s.
**Aucun échantillonnage n'est nécessaire** : les 3 algorithmes sont entraînés sur tout le train.
L'option `max_train_samples` (échantillon stratifié par station × mois, pour garder toutes les stations et saisons)
reste disponible pour le réentraînement si les ressources sont limitées (Kubeflow, sprint 3).

## 5. Résultats sur la validation contrôlée

| Modèle retenu | Réglage | Détection pics | Fausses alertes | AUC pics | Entraînement | Latence unitaire | Taille |
|---|---|---|---|---|---|---|---|
| Isolation Forest | 100 arbres, `max_samples` 2048 | 63,9 % | 1,15 % | 0,978 | 1 s | 11 à 15 ms | 9 Mo |
| LOF | 100 voisins, standard | 98,3 % | 3,35 % | 0,996 | 20 à 32 s | 11 à 25 ms | 96 Mo |
| One-Class SVM | `gamma` 0,1, robust | 96,4 % | 1,44 % | 0,994 | 52 s | 5 à 6 ms | 0,5 Mo |

Les taux sont identiques d'un lancement à l'autre (graine fixe) ; les temps varient selon la charge de la machine
(deux lancements, 12 cœurs).

Constats :
- **One-Class SVM** offre le meilleur compromis : presque autant de détection que LOF, avec 2,3 fois moins de fausses
  alertes, le modèle le plus léger et la latence la plus faible. Le réglage `gamma=scale` est presque équivalent
  (96,2 % / 1,40 %) et 3 fois plus rapide à entraîner.
- **LOF** détecte le plus, mais ses fausses alertes triplent entre le train (1 %) et la validation (3 à 10 % selon le
  voisinage) : il est très sensible à la dérive saisonnière (train sur une année, validation en automne). Il stocke
  aussi tout le train (96 Mo).
- **Isolation Forest** a peu de fausses alertes mais manque un pic sur trois à ce seuil, alors que son AUC (0,978) est
  bonne : ses pics sont bien classés mais son seuil à 1 % est trop strict pour eux.
- **Aucun modèle ne détecte le capteur figé (≤ 3 %) ni les valeurs impossibles (≤ 9 %)** : un modèle point par point
  ne voit pas une répétition, et une valeur impossible devient un NaN imputé « normal ». Ces cas relèvent des règles
  (`defaut_capteur`), ce qui confirme le choix d'une règle métier séparée du score (règle 6).
- Les 3 modèles respectent la latence maximale de 50 ms par mesure.

Ce classement est **provisoire** : le choix du modèle se fait sur le test contrôlé (`evaluation.py`), voir
[model_choice.md](model_choice.md). Attention : ici, les « fausses alertes » comptent toute alerte sur une ligne de
référence, y compris sur les vrais épisodes de pollution extrême. L'évaluation sur le test les exclut (fausses alertes
sur les heures ordinaires seulement), ce qui change le classement d'Isolation Forest.

## 6. Limites

- Les pics injectés sont nets (au-dessus du 99e centile) : les taux de détection sont optimistes par rapport à des
  anomalies réelles plus subtiles.
- La validation ne couvre que l'automne ; le test, que l'hiver. Les fausses alertes mesurées dépendent de la saison.
- Un seul tirage d'injection (graine 42) : pas d'intervalle de confiance.

## 7. Sauvegarde et traçabilité

- Pour chaque algorithme, le meilleur réglage est sauvegardé dans `models/<algorithme>/` (format MLflow) et comme
  artefact `model` de son run MLflow (tag `selected=true`).
- Format **skops** (par défaut dans MLflow 3.18) plutôt que pickle : le chargement refuse tout type non déclaré.
  Types déclarés : `SKOPS_TRUSTED_TYPES` dans `novaria.models.detectors`.
- Le code `novaria` est embarqué dans le modèle (`code_paths`) : l'API le charge sans installer le paquet.
- Signature : 17 variables **optionnelles** (une mesure manquante est imputée, pas refusée), sortie = score.
- Tags de chaque run : `git_sha` (suffixe `-dirty` si modifications non commitées), `dataset_sha256`, `algorithm` ;
  sur le run retenu, `model_sha256` (empreinte du fichier `model.skops`, à vérifier par l'API au chargement).
- MLflow : `MLFLOW_TRACKING_URI` si définie (serveur Docker Compose), sinon suivi local `mlflow.db` + `mlartifacts/`.
- L'enregistrement dans le Registry (`air-anomaly-detector`, alias `@champion` / `@previous`) est fait par `promotion.py`.
