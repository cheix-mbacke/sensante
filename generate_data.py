import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)

np.random.seed(42)
n = 500

# Proportions des diagnostics
diagnostics = np.random.choice(
    ["sain", "paludisme", "grippe", "typhoide"],
    n,
    p=[0.316, 0.272, 0.260, 0.152]
)

# Temperature selon le diagnostic
temperature = []
for d in diagnostics:
    if d == "grippe":
        temperature.append(round(np.random.normal(38.5, 0.5), 1))
    elif d == "paludisme":
        temperature.append(round(np.random.normal(39.4, 0.4), 1))
    elif d == "typhoide":
        temperature.append(round(np.random.normal(39.0, 0.4), 1))
    else:  # sain
        temperature.append(round(np.random.normal(36.8, 0.3), 1))

df = pd.DataFrame({
    "age":             np.random.randint(18, 70, n),
    "sexe":            np.random.choice(["M", "F"], n),
    "temperature":     temperature,
    "tension_sys":     np.random.randint(10, 14, n),
    "toux":            np.random.randint(0, 2, n),
    "fatigue":         np.random.randint(0, 2, n),
    "maux_tete":       np.random.randint(0, 2, n),
    "courbatures":     np.random.randint(0, 2, n),
    "frissons":        np.random.randint(0, 2, n),
    "perte_appetit":   np.random.randint(0, 2, n),
    "diagnostic":      diagnostics,
})

df.to_csv("data/patients_dakar.csv", index=False)
print(f"Dataset cree : {n} patients dans data/patients_dakar.csv")