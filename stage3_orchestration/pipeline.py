"""Reusable ML-pipeline building blocks for the Titanic survival model."""
import os
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
# Dataset lives in ../data (repo layout). TITANIC_DATA overrides it; ./data is the fallback (Airflow dags folder).
DATA_FILE = Path(os.getenv("TITANIC_DATA", HERE.parent / "data" / "Titanic-Dataset.csv"))
if not DATA_FILE.exists() and (HERE / "data" / "Titanic-Dataset.csv").exists():
    DATA_FILE = HERE / "data" / "Titanic-Dataset.csv"

NUMERIC = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL = ["Sex", "Embarked"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Survived"


def load_data(path=DATA_FILE):
    """Stage: data ingestion."""
    return pd.read_csv(path)


def build_preprocessor():
    """Stage: preprocessing (impute missing values, scale numbers, one-hot encode text)."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])


def build_pipeline(model=None):
    """Stage: preprocessing + model chained into ONE object (no train/serve skew)."""
    if model is None:
        model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    return Pipeline([("prep", build_preprocessor()), ("model", model)])
