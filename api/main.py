# api/main.py
# API FastAPI pour SenSante - Assistant pré-diagnostic médical

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("Client Groq initialisé.")
    
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

# --- Création de l'application ---
app = FastAPI(
    title="SenSante API",
    description="Assistant pré-diagnostic médical pour le Sénégal",
    version="0.2.0"
)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # En dev : tout accepter
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Route de santé ---
@app.get("/health")
def health_check():
    """Vérification de l'état de l'API."""
    return {
        "status": "ok",
        "message": "SenSante API is running"
    }

# --- Schémas Pydantic ---

class PatientInput(BaseModel):
    """Données d'entrée : les symptômes d'un patient."""

    age: int = Field(..., ge=0, le=120, description="Âge en années")
    sexe: str = Field(..., description="Sexe : M ou F")
    temperature: float = Field(..., ge=35.0, le=42.0, description="Température en Celsius")
    tension_sys: int = Field(..., ge=60, le=250, description="Tension systolique")
    toux: bool = Field(..., description="Présence de toux")
    fatigue: bool = Field(..., description="Présence de fatigue")
    maux_tete: bool = Field(..., description="Présence de maux de tête")
    region: str = Field(..., description="Région du Sénégal")


class DiagnosticOutput(BaseModel):
    """Données de sortie : résultat du diagnostic."""

    diagnostic: str
    probabilite: float
    confiance: str
    message: str


# --- Chargement du modèle ---
print("Chargement du modèle...")

model = joblib.load("models/model.pkl")
le_sexe = joblib.load("models/encoder_sexe.pkl")
le_region = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

print(f"Modèle chargé : {type(model).__name__}")
print(f"Classes : {list(model.classes_)}")


# --- Endpoint de prédiction ---
@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):

    # 1. Encodage sexe
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message="Sexe invalide : utiliser M ou F"
        )

    # 2. Encodage région
    try:
        region_enc = le_region.transform([patient.region])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Région inconnue : {patient.region}"
        )

    # 3. Construction features
    features = np.array([[
        patient.age,
        sexe_enc,
        patient.temperature,
        patient.tension_sys,
        int(patient.toux),
        int(patient.fatigue),
        int(patient.maux_tete),
        region_enc
    ]])

    # 4. Prédiction
    diagnostic = model.predict(features)[0]
    probas = model.predict_proba(features)[0]
    proba_max = float(probas.max())

    # 5. Confiance
    if proba_max >= 0.7:
        confiance = "haute"
    elif proba_max >= 0.4:
        confiance = "moyenne"
    else:
        confiance = "faible"

    # 6. Messages
    messages = {
        "palu": "Suspicion de paludisme. Consultez un médecin rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation recommandés.",
        "typh": "Suspicion de typhoïde. Consultation médicale nécessaire.",
        "sain": "Pas de pathologie détectée. Continuez à surveiller."
    }

    # 7. Retour
    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un médecin.")
    )
    
class ExplainInput(BaseModel):
    diagnostic: str
    probabilite: float
    age: int
    sexe: str
    temperature: float
    region: str

class ExplainOutput(BaseModel):
    explication: str
    modele_llm: str = "llama-3.1-8b-instant"

# SYSTEM_PROMPT est EN DEHORS de la classe, au niveau du module
SYSTEM_PROMPT = """Tu es un assistant médical sénégalais.
Explique le résultat en français simple.
Sois rassurant mais recommande une consultation.
Maximum 3 phrases. Ne fais JAMAIS de diagnostic."""

@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):
    if not groq_client:
        return ExplainOutput(
            explication="Service indisponible.",
            modele_llm="aucun"
        )
    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, "
        f"région {data.region}\n"
        f"Température : {data.temperature}°C\n"
        f"Diagnostic : {data.diagnostic} "
        f"({data.probabilite:.0%})\n"
        f"Explique ce résultat au patient."
    )
    try:
        r = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt}
            ],
            max_tokens=200, temperature=0.3
        )
        explication = r.choices[0].message.content
    except Exception as e:
        explication = f"Erreur LLM : {str(e)}"
    return ExplainOutput(explication=explication)