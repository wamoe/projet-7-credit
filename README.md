# API de Scoring Crédit (Flask)

Cette API REST, développée avec **Flask**, permet d’interroger en temps réel un modèle de **Machine Learning** afin d’évaluer le **risque de défaut** d’un client.

- **Endpoint principal** : `POST /predict`
- **Sortie** : probabilité de défaut + décision métier (**Accordé** / **Refusé**) selon un **seuil** paramétré dans le code.

---

## Sommaire

- [Fonctionnalités](#fonctionnalités)
- [Structure du projet](#structure-du-projet)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Lancer l’API](#lancer-lapi)
- [Documentation des endpoints](#documentation-des-endpoints)
- [Exemples d’appels](#exemples-dappels)
- [Robustesse et gestion des erreurs](#robustesse-et-gestion-des-erreurs)
- [Dépannage](#dépannage)

---

## Fonctionnalités

- **Chargement automatique** du modèle sérialisé (`joblib`)
- **Alignement des features** : réordonne et complète les colonnes exactement comme à l’entraînement
- **Tolérance aux colonnes manquantes** : les features absentes sont remplies avec `0`
- **Décision métier** à partir d’un **seuil** (ex. `0.45`)
- **Health check** pour vérifier l’état du modèle

---

## Structure du projet

La structure attendue (minimale) :

```
project_root/
├─ src/
│  └─ app.py
└─ model_production/
   ├─ model.pkl
   └─ features.csv
```

- `model_production/model.pkl` : modèle entraîné (doit exposer `predict_proba`)
- `model_production/features.csv` : liste ordonnée des variables attendues (la **première colonne** doit contenir les noms de features, dans l’ordre d’entraînement)

---

## Prérequis

- **Python 3.8+**
- Un fichier `requirements.txt` (dépendances Python)

> ✅ Recommandé : utiliser un environnement virtuel (venv/conda) pour isoler les dépendances.

---

## Installation

Depuis la racine du projet :

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Lancer l’API

L’API est conçue pour être lancée depuis la racine du projet.

```bash
python src/app.py
```

Au démarrage, vous devriez voir un log similaire :

```
Chemin du modèle détecté : .../model_production/model.pkl
Chargement du modèle...
Modèle chargé avec succès.
 * Running on http://0.0.0.0:5000/
```

Par défaut :
- **Host** : `0.0.0.0`
- **Port** : `5000`
- **Mode debug** : activé (utile en dev, à désactiver en prod)

---

## Documentation des endpoints

### 1) Health Check

Vérifie que l’API répond et indique si le modèle est chargé.

- **URL** : `/`
- **Méthode** : `GET`
- **Réponse** : HTML simple

Exemple :

```bash
curl http://127.0.0.1:5000/
```

---

### 2) Prédiction

Envoie des données client(s) et renvoie un score.

- **URL** : `/predict`
- **Méthode** : `POST`
- **Content-Type** : `application/json`

#### Format d’entrée

 **Format recommandé** : une **liste d’objets** (même pour un seul client)

```json
[
  {
    "EXT_SOURCE_1": 0.5,
    "EXT_SOURCE_2": 0.6,
    "EXT_SOURCE_3": 0.4,
    "PAYMENT_RATE": 0.05,
    "DAYS_BIRTH": -15000,
    "AMT_CREDIT": 200000
  }
]
```

>  Remarque : l’implémentation actuelle construit le DataFrame via `pd.DataFrame(data)`.
> En pratique, **envoyer une liste** est le format le plus robuste.

#### Format de sortie

Exemple de réponse :

```json
{
  "probability": 0.7823,
  "threshold": 0.45,
  "prediction": 1,
  "status": "Refusé"
}
```

Champs :
- `prediction` : `0` (pas de défaut) ou `1` (défaut)
- `probability` : probabilité de défaut (entre `0` et `1`)
- `threshold` : seuil de décision appliqué (ici `0.45`)
- `status` : décision métier (**Accordé** si `probability < threshold`, sinon **Refusé**)

---

## Exemples d’appels

### cURL

```bash
curl -X POST "http://127.0.0.1:5000/predict"   -H "Content-Type: application/json"   -d '[
    {
      "EXT_SOURCE_1": 0.5,
      "EXT_SOURCE_2": 0.6,
      "EXT_SOURCE_3": 0.4,
      "PAYMENT_RATE": 0.05,
      "DAYS_BIRTH": -15000,
      "AMT_CREDIT": 200000
    }
  ]'
```

### Python (requests)

```python
import requests

url = "http://127.0.0.1:5000/predict"
payload = [
    {
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.6,
        "EXT_SOURCE_3": 0.4,
        "PAYMENT_RATE": 0.05,
        "DAYS_BIRTH": -15000,
        "AMT_CREDIT": 200000
    }
]

r = requests.post(url, json=payload, timeout=10)
print(r.status_code)
print(r.json())
```

---

## Robustesse et gestion des erreurs

### Alignement strict des colonnes

- L’API charge `features.csv` au démarrage.
- À la prédiction :
  - les colonnes sont **réordonnées** selon l’ordre d’entraînement
  - les colonnes manquantes sont ajoutées avec `fill_value=0`
  - les colonnes “en trop” dans le JSON sont ignorées

### Erreurs possibles

- **Modèle non chargé** : retourne `500` avec un message explicite
- **JSON invalide / erreur de traitement** : retourne `500` avec le détail de l’erreur

---

## Dépannage

### “Le fichier n’existe pas : .../model_production/model.pkl”
- Vérifier la présence de :
  - `model_production/model.pkl`
  - `model_production/features.csv`
- Vérifier que vous lancez bien :
  ```bash
  python src/app.py
  ```

### “Le modèle n'est pas chargé. Vérifiez les logs du serveur.”
- L’API a démarré mais le chargement a échoué.
- Lire le message d’erreur affiché au lancement (terminal).

### Le endpoint `/predict` renvoie une erreur avec un JSON “unique”
- Utilisez le format **liste** :
  ```json
  [ { ... } ]
  ```

---

## Notes “Production”

- Désactiver `debug=True`
- Placer l’API derrière un serveur WSGI (ex : gunicorn)
- Mettre en place :
  - authentification
  - limitation de débit (rate limiting)
  - logs structurés
  - monitoring

---

**Auteur** :  
**Licence** : 
