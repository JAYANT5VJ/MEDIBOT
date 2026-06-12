import sqlite3
from pathlib import Path
from typing import Optional
import pandas as pd

DB_PATH = Path("reviews.db")

def _connect():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    con = _connect()
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS drug_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        drug_name TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('User', 'Physician')),
        reviewer_name TEXT,
        review_text TEXT NOT NULL,
        sentiment_score REAL NOT NULL,
        sentiment_label TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)
    con.commit()
    con.close()

def insert_review(
    drug_name: str,
    role: str,
    reviewer_name: Optional[str],
    review_text: str,
    sentiment_score: float,
    sentiment_label: str,
    created_at_iso: str
):
    con = _connect()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO drug_reviews
        (drug_name, role, reviewer_name, review_text, sentiment_score, sentiment_label, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (drug_name, role, reviewer_name, review_text, sentiment_score, sentiment_label, created_at_iso))
    con.commit()
    con.close()

def load_reviews_df() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame(columns=[
            "id","drug_name","role","reviewer_name","review_text",
            "sentiment_score","sentiment_label","created_at"
        ])
    con = _connect()
    df = pd.read_sql_query("SELECT * FROM drug_reviews ORDER BY id DESC", con)
    con.close()
    return df

def delete_review(review_id: int):
    con = _connect()
    cur = con.cursor()
    cur.execute("DELETE FROM drug_reviews WHERE id = ?", (review_id,))
    con.commit()
    con.close()

def update_review(review_id: int, new_text: str, sentiment_score: float, sentiment_label: str):
    con = _connect()
    cur = con.cursor()
    cur.execute("""
        UPDATE drug_reviews
        SET review_text = ?, sentiment_score = ?, sentiment_label = ?
        WHERE id = ?
    """, (new_text, sentiment_score, sentiment_label, review_id))
    con.commit()
    con.close()