from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

@dataclass
class TrainResult:
    model: Pipeline
    accuracy: float
    report: str
    confusion: list

def train_text_model(df: pd.DataFrame, text_col: str, label_col: str) -> TrainResult:
    df = df.dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(str)

    X = df[text_col].values
    y = df[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=50000)),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    acc = float(accuracy_score(y_test, preds))
    rep = classification_report(y_test, preds)
    cm = confusion_matrix(y_test, preds).tolist()

    return TrainResult(model=pipe, accuracy=acc, report=rep, confusion=cm)

def predict_text(model: Pipeline, text: str) -> str:
    return str(model.predict([text])[0])