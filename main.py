import pandas as pd
import random

# Base vocab for realistic medical fields
uses_list = [
    "Pain relief", "Fever reduction", "Bacterial infections", "Hypertension",
    "Type 2 diabetes", "High cholesterol", "Asthma", "Allergic rhinitis",
    "Depression", "Anxiety disorders", "Gastric ulcer", "GERD",
    "Arthritis inflammation", "Migraine", "Epilepsy", "Heart failure",
    "Chronic kidney disease", "Thyroid disorders", "Dermatitis", "Acne",
    "Fungal infections", "Viral infections", "COPD", "Stroke prevention",
    "Anticoagulation", "Immunosuppression", "Chemotherapy support"
]

side_effects_list = [
    "Nausea", "Vomiting", "Headache", "Dizziness", "Dry mouth",
    "Diarrhea", "Constipation", "Rash", "Fatigue", "Insomnia",
    "Drowsiness", "Abdominal pain", "Muscle pain", "Weight gain",
    "Photosensitivity", "Blurred vision", "Tremor", "Palpitations"
]

warnings_list = [
    "Avoid in pregnancy unless prescribed",
    "Caution in liver impairment",
    "Caution in renal impairment",
    "May interact with alcohol",
    "Monitor blood pressure",
    "Risk of bleeding with anticoagulants",
    "Avoid overdose",
    "May cause drowsiness; avoid driving",
    "Check for drug–drug interactions",
    "Use lowest effective dose",
    "Monitor blood glucose levels",
    "Not for pediatric use without advice",
    "Hypersensitivity reactions possible"
]

drug_prefix = [
    "Para", "Ibu", "Amoxi", "Metro", "Atorva", "Levo", "Cef", "Azi", "Dolo",
    "Gaba", "Flu", "Cipro", "Clari", "Rani", "Ome", "Pant", "Rosu",
    "Telmi", "Amlodi", "Metfor", "Glime", "Insu", "Sertra", "Escita",
    "Alpra", "Diaze", "Mont", "Fexo", "Lora", "Ceti", "Hydro"
]

drug_suffix = [
    "cillin", "fen", "pril", "sartan", "statin", "azole", "mycin",
    "dazole", "zepam", "xetine", "formin", "gliptin", "gliflozin",
    "barbital", "caine", "olol", "pine", "vir", "mab", "nib"
]

def random_drug_name():
    return random.choice(drug_prefix) + random.choice(drug_suffix)

rows = []
seen = set()

while len(rows) < 300:
    name = random_drug_name()
    if name in seen:
        continue
    seen.add(name)

    uses = random.sample(uses_list, k=random.randint(1, 2))
    side_effects = random.sample(side_effects_list, k=random.randint(2, 4))
    warnings = random.sample(warnings_list, k=random.randint(1, 2))

    rows.append({
        "drug_name": name,
        "uses": ", ".join(uses),
        "side_effects": ", ".join(side_effects),
        "warnings": ", ".join(warnings)
    })

df = pd.DataFrame(rows)
df = df.sort_values("drug_name").reset_index(drop=True)

df.to_csv("data/drug_info.csv", index=False)

print("✅ Generated 1000 drug records → data/drug_info.csv")