"""
main.py — API FastAPI de scoring crédit

Pour lancer l'API :
    uvicorn main:app --reload --port 8000

Documentation automatique disponible sur :
    http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.schemas import ClientData, PredictionResponse
from api.model import predire

# ── Création de l'application FastAPI ─────────────────────────────────────────
app = FastAPI(
    title="Home Credit Scoring API",
    description="""
    API de scoring pour la prédiction du risque de défaut de paiement.
    
    ## Comment utiliser
    
    Envoyez les données d'un client sur `/predict` et recevez :
    - La probabilité de défaut (entre 0 et 1)
    - La décision (ACCORDÉ ou REFUSÉ)
    - Le niveau de risque (FAIBLE, MODÉRÉ, ÉLEVÉ, TRÈS ÉLEVÉ)
    
    ## Modèle utilisé
    XGBoost — AUC-ROC : 0.7476 — Seuil optimal : 0.75
    """,
    version="1.0.0"
)

# ── Configuration CORS ────────────────────────────────────────────────────────
# Permet à Streamlit (et autres) d'appeler l'API depuis un autre port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Route 1 : Vérification que l'API fonctionne ───────────────────────────────
@app.get("/")
def accueil():
    """
    Route de base — vérifie que l'API est en ligne.
    """
    return {
        "message"   : "Home Credit Scoring API — opérationnelle ✅",
        "version"   : "1.0.0",
        "modele"    : "XGBoost",
        "seuil"     : 0.75,
        "docs"      : "/docs"
    }


# ── Route 2 : Vérification santé de l'API ────────────────────────────────────
@app.get("/health")
def health_check():
    """
    Route utilisée par les systèmes de monitoring pour vérifier
    que l'API répond correctement.
    """
    return {"status": "healthy"}


# ── Route 3 : Prédiction ──────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
def predict(client: ClientData):
    """
    Prédit le risque de défaut de paiement pour un client.

    Envoie les données du client et reçois :
    - score_defaut : probabilité de défaut (0 à 1)
    - decision : ACCORDÉ ou REFUSÉ
    - niveau_risque : FAIBLE / MODÉRÉ / ÉLEVÉ / TRÈS ÉLEVÉ
    - seuil_utilise : seuil de décision (0.75)
    - message : explication de la décision
    """
    try:
        # Convertir les données Pydantic en dictionnaire
        data = client.model_dump()

        # Faire la prédiction
        resultat = predire(data)

        return resultat

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction : {str(e)}"
        )


# ── Lancement direct ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
