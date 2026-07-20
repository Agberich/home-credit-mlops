"""
model.py — Chargement et utilisation du modèle XGBoost

Ce fichier s'occupe uniquement de :
1. Charger le modèle depuis le fichier .pkl
2. Préparer les données reçues
3. Faire la prédiction
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Seuil optimal trouvé pendant la modélisation
SEUIL_OPTIMAL = 0.75

# Chemin vers le modèle sauvegardé
MODEL_PATH = Path(__file__).parent.parent / 'models' / 'XGBoost.pkl'

# Chargement du modèle au démarrage de l'API
# (on le charge une seule fois, pas à chaque requête)
try:
    model = joblib.load(MODEL_PATH)
    print(f'✅ Modèle chargé depuis : {MODEL_PATH}')
except Exception as e:
    print(f'❌ Erreur chargement modèle : {e}')
    model = None


def preparer_features(data: dict) -> pd.DataFrame:
    """
    Transforme les données brutes du client en features
    utilisées par le modèle.

    On recréé les mêmes features que dans le notebook 02.
    """
    # Créer un DataFrame avec les données reçues
    df = pd.DataFrame([data])

    # Recréer les features engineered (comme dans notebook 02)
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365.25
    df['YEARS_EMPLOYED'] = -df['DAYS_EMPLOYED'] / 365.25
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1)
    df['EXT_SOURCE_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['YEARS_REGISTRATION'] = -df['DAYS_REGISTRATION'] / 365.25
    df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)

    return df


def predire(data: dict) -> dict:
    """
    Fait la prédiction pour un client.

    Paramètre : data → dictionnaire avec les données du client
    Retourne  : dictionnaire avec le score et la décision
    """
    if model is None:
        raise Exception("Modèle non chargé")

    # Préparer les features
    df = preparer_features(data)

    # Garder uniquement les colonnes que le modèle connaît
    # (celles utilisées pendant l'entraînement)
    feature_cols = model.get_booster().feature_names
    
    # Ajouter les colonnes manquantes avec 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    # Garder uniquement les colonnes du modèle dans le bon ordre
    df = df[feature_cols]

    # Prédiction — probabilité de défaut
    proba = model.predict_proba(df)[0][1]

    # Décision selon le seuil optimal
    decision = "REFUSÉ" if proba >= SEUIL_OPTIMAL else "ACCORDÉ"

    # Niveau de risque
    if proba < 0.25:
        niveau = "FAIBLE"
    elif proba < 0.50:
        niveau = "MODÉRÉ"
    elif proba < 0.75:
        niveau = "ÉLEVÉ"
    else:
        niveau = "TRÈS ÉLEVÉ"

    # Message explicatif
    if decision == "ACCORDÉ":
        message = f"Crédit accordé. Probabilité de défaut : {proba*100:.1f}% (sous le seuil de {SEUIL_OPTIMAL*100:.0f}%)"
    else:
        message = f"Crédit refusé. Probabilité de défaut : {proba*100:.1f}% (au-dessus du seuil de {SEUIL_OPTIMAL*100:.0f}%)"

    return {
        "score_defaut"  : round(float(proba), 4),
        "decision"      : decision,
        "niveau_risque" : niveau,
        "seuil_utilise" : SEUIL_OPTIMAL,
        "message"       : message
    }
