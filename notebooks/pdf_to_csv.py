import pdfplumber
import pandas as pd
import re

rows = []

with pdfplumber.open("data/patients_dakar.pdf") as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        for line in text.split("\n"):
            line = line.strip()
            # Ignorer l'entête
            if line.startswith("age") or line == "":
                continue
            # Séparer age et sexe collés ex: "18M 38.7..."
            match = re.match(r"(\d+)([MF])\s+([\d.]+)\s+(\d+)(\d)(\d)(\d)(\d)(\d)(\d)\s+(\S+)\s+(\S+)", line)
            if match:
                age, sexe, temp, tension, toux, fatigue, maux_tete, frissons, nausee, col10, region, diagnostic = match.groups()
                rows.append([age, sexe, temp, tension, toux, fatigue, maux_tete, frissons, nausee, region, diagnostic])

df = pd.DataFrame(rows, columns=[
    "age", "sexe", "temperature", "tension_sys",
    "toux", "fatigue", "maux_tete", "frissons", "nausee",
    "region", "diagnostic"
])

df.to_csv("data/patients_dakar.csv", index=False)
print(f"CSV cree : {len(df)} patients")
print(df.head())