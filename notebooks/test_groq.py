import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system",
         "content": "Tu es un assistant médical sénégalais. "
                    "Réponds en français simple. Maximum 3 phrases."},
        {"role": "user",
         "content": "Quels sont les symptômes du paludisme ?"}
    ],
    max_tokens=200,
    temperature=0.3
)

print(response.choices[0].message.content)
print(f"Tokens utilisés : {response.usage.total_tokens}")

# Test avec le format SénSanté
response2 = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system",
         "content": """Tu es un assistant médical sénégalais.
Tu reçois un diagnostic et des données patient.
Explique le résultat en français simple,
comme un médecin parlerait à son patient.
Sois rassurant mais recommande une consultation.
Maximum 3 phrases.
Ne fais JAMAIS de diagnostic toi-même."""},
        {"role": "user",
         "content": """Patient : Femme, 28 ans, région Dakar
Symptômes : température 39.5, toux, fatigue, maux de tête
Diagnostic du modèle : paludisme (probabilité 72%)
Explique ce résultat au patient."""}
    ],
    max_tokens=200,
    temperature=0.3
)

print("=== Explication SénSanté ===")
print(response2.choices[0].message.content)