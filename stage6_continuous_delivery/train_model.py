"""Train the Titanic survival model; writes titanic_model.pkl, metrics.json, test_predictions.csv."""
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
DATA_FILE = Path(os.getenv("TITANIC_DATA", HERE.parent / "data" / "Titanic-Dataset.csv"))

NUMERIC = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL = ["Sex", "Embarked"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Survived"


def build_pipeline():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])
    return Pipeline([("prep", prep),
                     ("model", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42))])


def main():
    df = pd.read_csv(DATA_FILE)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "dataset": DATA_FILE.name,
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
    }

    joblib.dump(pipe, HERE / "titanic_model.pkl")
    (HERE / "metrics.json").write_text(json.dumps(metrics, indent=2))
    out = X_test.copy()
    out["actual"] = y_test
    out["predicted"] = pred
    out.to_csv(HERE / "test_predictions.csv", index=False)

    print("Model saved: titanic_model.pkl")
    print("Metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
