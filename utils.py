from pathlib import Path
import pandas as pd

DRUG_INFO_PATH = Path("drug_info.csv")

def load_drug_info() -> pd.DataFrame:
    if not DRUG_INFO_PATH.exists():
        return pd.DataFrame(columns=["drug_name", "uses", "side_effects", "warnings"])
    df = pd.read_csv(DRUG_INFO_PATH)
    # normalize names
    df["drug_name"] = df["drug_name"].astype(str).str.strip()
    return df

def top_n_drugs_from_reviews(reviews_df: pd.DataFrame, fallback_list: list[str], n: int = 30) -> list[str]:
    if reviews_df.empty:
        return fallback_list[:n]
    counts = reviews_df["drug_name"].value_counts()
    drugs = counts.index.tolist()
    # include fallback to reach n
    for d in fallback_list:
        if d not in drugs:
            drugs.append(d)
        if len(drugs) >= n:
            break
    return drugs[:n]