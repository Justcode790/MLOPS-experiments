"""Plain-Python pipeline tasks. The SAME functions are orchestrated by Prefect (flow.py)
and by Apache Airflow (airflow_dag.py). Tasks hand data to each other through files in ./work."""
import json
import os
import shutil

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from pipeline import DATA_FILE, FEATURES, HERE, TARGET, build_pipeline

WORK = HERE / "work"
OUTPUT = HERE / "output"
MIN_ACCURACY = float(os.getenv("MIN_ACCURACY", "0.75"))
MIN_F1 = float(os.getenv("MIN_F1", "0.65"))


def ingest_and_validate():
    """Task 1: read the CSV and run data-quality checks."""
    WORK.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_FILE)
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    if len(df) < 800:
        raise ValueError(f"Too few rows: {len(df)}")
    if set(df[TARGET].unique()) - {0, 1}:
        raise ValueError("Target must be 0/1")
    if df[FEATURES].isna().mean().max() > 0.30:
        raise ValueError("A feature has more than 30% missing values")
    df.to_csv(WORK / "clean.csv", index=False)
    summary = {"rows": len(df), "survival_rate": round(float(df[TARGET].mean()), 3)}
    print("Data validation passed:", summary)
    return summary


def split_data():
    """Task 2: stratified train/test split."""
    df = pd.read_csv(WORK / "clean.csv")
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[TARGET])
    train.to_csv(WORK / "train.csv", index=False)
    test.to_csv(WORK / "test.csv", index=False)
    print(f"Split: train={len(train)} test={len(test)}")
    return {"train": len(train), "test": len(test)}


def train_model():
    """Task 3: fit preprocessing + model pipeline."""
    train = pd.read_csv(WORK / "train.csv")
    pipe = build_pipeline()
    pipe.fit(train[FEATURES], train[TARGET])
    joblib.dump(pipe, WORK / "titanic_model.pkl")
    print("Model trained")
    return "trained"


def evaluate_model():
    """Task 4: evaluate on the hold-out set."""
    test = pd.read_csv(WORK / "test.csv")
    pipe = joblib.load(WORK / "titanic_model.pkl")
    pred = pipe.predict(test[FEATURES])
    proba = pipe.predict_proba(test[FEATURES])[:, 1]
    y = test[TARGET]
    metrics = {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
    }
    (WORK / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Metrics:", metrics)
    return metrics


def quality_gate():
    """Task 5: stop the pipeline if the model is not good enough."""
    m = json.loads((WORK / "metrics.json").read_text())
    if m["accuracy"] < MIN_ACCURACY or m["f1"] < MIN_F1:
        raise ValueError(f"QUALITY GATE FAILED: {m}")
    print("QUALITY GATE PASSED")
    return "passed"


def log_to_mlflow():
    """Task 6: record the run in MLflow and register the model."""
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(f"sqlite:///{(HERE / 'mlflow.db').as_posix()}")
    name = "titanic-orchestrated"
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(HERE / "mlartifacts").as_uri())
    mlflow.set_experiment(name)

    metrics = json.loads((WORK / "metrics.json").read_text())
    pipe = joblib.load(WORK / "titanic_model.pkl")
    with mlflow.start_run(run_name="orchestrated-run") as run:
        mlflow.set_tag("orchestrator", "prefect-or-airflow")
        mlflow.log_params({"algorithm": "RandomForestClassifier", "n_estimators": 200, "max_depth": 6})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")
    mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", "titanic-survival-model-orchestrated")
    MlflowClient().set_registered_model_alias("titanic-survival-model-orchestrated", "champion", mv.version)
    print(f"Registered model version {mv.version}")
    return mv.version


def publish_model():
    """Task 7: copy the validated model + metrics to ./output (the deliverable)."""
    OUTPUT.mkdir(exist_ok=True)
    shutil.copy(WORK / "titanic_model.pkl", OUTPUT / "titanic_model.pkl")
    shutil.copy(WORK / "metrics.json", OUTPUT / "metrics.json")
    print("Published to", OUTPUT)
    return str(OUTPUT)
