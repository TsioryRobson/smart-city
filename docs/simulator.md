# Simulateur IoT

Installez les dépendances du dépôt et placez les CSV [Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijingmultisiteairqualitydata) dans `data/raw/`.

```bash
python -m novaria.simulator --limit 3
python -m novaria.simulator --station Changping --sensor IoT-003 --interval 0.5 --limit 10
python -m novaria.simulator --transport http --url http://127.0.0.1:8000/predict --scenario pic --limit 3
python -m pytest -q tests/simulator
```

Le simulateur lit le CSV brut de la station choisie via la préparation partagée. Il ne rejoue que les lignes à partir du 1er mars 2016, date de début de la période de démonstration. Le dataset se termine le 28 février 2017. Chaque événement a son propre identifiant et conserve l'horodatage historique en heure locale de Pékin. `--csv` sélectionne un fichier, `--interval` détermine le délai entre envois, et `--limit` limite le nombre de mesures. Ctrl+C et SIGTERM arrêtent le processus.

| Valeur de `--scenario` | Effet sur la première émission |
|---|---|
| `reference` | Mesure historique telle que fournie, avec ses éventuels manquants |
| `pic` | Pic plausible de PM2.5 et de PM10 |
| `valeur_impossible` | PM2.5 négatif |
| `combinaison_inhabituelle` | Peu de particules et plusieurs gaz à des niveaux élevés |
| `valeur_manquante` | PM2.5 envoyé à `null` |

Sous WSL ou Linux, `kill -USR1 <PID>` applique le scénario sélectionné à la prochaine émission. Avec `reference`, ce signal injecte un pic. En console, `injection` figure dans le JSON. En HTTP, le simulateur journalise l'identifiant et le scénario sur stderr, transmet uniquement les champs de `/predict` et affiche la réponse sur stdout.

Un futur transport Kafka implémentera `Transport.send()` et publiera le schéma `RawMeasurement`, étiquette comprise, sur `mesures.brutes`.
