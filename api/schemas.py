"""
schemas.py — Structure des données attendues par l'API

Ce fichier définit exactement quelles données le client doit envoyer
et ce que l'API va retourner.

Pydantic vérifie automatiquement que les données reçues sont correctes.
Si une donnée est manquante ou du mauvais type → erreur claire.
"""

from pydantic import BaseModel
from typing import Optional


class ClientData(BaseModel):
    """
    Données d'un client envoyées à l'API pour scoring.
    On utilise les features les plus importantes identifiées par SHAP.
    """

    # Scores externes (features les plus prédictives)
    EXT_SOURCE_1: Optional[float] = 0.5
    EXT_SOURCE_2: Optional[float] = 0.5
    EXT_SOURCE_3: Optional[float] = 0.5

    # Informations financières
    AMT_CREDIT: float           # Montant du crédit demandé
    AMT_INCOME_TOTAL: float     # Revenu annuel du client
    AMT_ANNUITY: Optional[float] = 0.0     # Montant de l'annuité
    AMT_GOODS_PRICE: Optional[float] = 0.0 # Prix du bien financé

    # Informations personnelles
    DAYS_BIRTH: float           # Âge en jours (négatif)
    DAYS_EMPLOYED: Optional[float] = -1000  # Ancienneté emploi en jours

    # Autres features importantes
    REGION_RATING_CLIENT: Optional[int] = 2
    DAYS_ID_PUBLISH: Optional[float] = -2000
    DAYS_REGISTRATION: Optional[float] = -3000
    DAYS_LAST_PHONE_CHANGE: Optional[float] = -500
    CNT_CHILDREN: Optional[int] = 0
    CNT_FAM_MEMBERS: Optional[float] = 2.0

    class Config:
        # Exemple affiché dans la documentation automatique de FastAPI
        json_schema_extra = {
            "example": {
                "EXT_SOURCE_1": 0.45,
                "EXT_SOURCE_2": 0.60,
                "EXT_SOURCE_3": 0.55,
                "AMT_CREDIT": 500000,
                "AMT_INCOME_TOTAL": 150000,
                "AMT_ANNUITY": 25000,
                "AMT_GOODS_PRICE": 450000,
                "DAYS_BIRTH": -12000,
                "DAYS_EMPLOYED": -2000,
                "REGION_RATING_CLIENT": 2,
                "DAYS_ID_PUBLISH": -2500,
                "DAYS_REGISTRATION": -4000,
                "DAYS_LAST_PHONE_CHANGE": -300,
                "CNT_CHILDREN": 1,
                "CNT_FAM_MEMBERS": 3.0
            }
        }


class PredictionResponse(BaseModel):
    """
    Réponse retournée par l'API après la prédiction.
    """
    score_defaut: float        # Probabilité de défaut (entre 0 et 1)
    decision: str              # "ACCORDÉ" ou "REFUSÉ"
    niveau_risque: str         # "FAIBLE", "MODÉRÉ", "ÉLEVÉ", "TRÈS ÉLEVÉ"
    seuil_utilise: float       # Seuil de décision (0.75)
    message: str               # Message explicatif
