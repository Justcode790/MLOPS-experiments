"""Stage 2 - MLflow: experiment tracking, model comparison and model registry."""
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow import MlflowClient
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split

from pipeline import FEATURES, HERE, TARGET, build_pipeline, load_data

EXPERIMENT = "titanic-survival"
REGISTERED_MODEL = "titanic-survival-model"

mlflow.set_tracking_uri(f"sqlite:///{(HERE / 'mlflow.db').as_posix()}")
if mlflow.get_experiment_by_name(EXPERIMENT) is None:
    mlflow.create_experiment(EXPERIMENT, artifact_location=(HERE / "mlartifacts").as_uri())
mlflow.set_experiment(EXPERIMENT)

# Candidate models = different hyper-parameters / algorithms to compare
CANDIDATES = {
    "logreg_C1": LogisticRegression(C=1.0, max_iter=1000),
    "logreg_C0.1": LogisticRegression(C=0.1, max_iter=1000),
    "random_forest_d4": RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42),
    "random_forest_d6": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42),
}

df = load_data()
X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

results = []
for name, estimator in CANDIDATES.items():
    with mlflow.start_run(run_name=name) as run:
        pipe = build_pipeline(estimator)

        # ---- log parameters ----
        mlflow.log_param("algorithm", type(estimator).__name__)
        mlflow.log_params({f"model__{k}": v for k, v in estimator.get_params().items()
                           if k in ("C", "n_estimators", "max_depth", "learning_rate")})
        mlflow.log_param("test_size", 0.2)
        mlflow.set_tags({"dataset": "Titanic-Dataset.csv", "stage": "experimentation"})

        # ---- train + evaluate ----
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba),
            "cv_accuracy_mean": cross_val_score(build_pipeline(estimator), X_train, y_train, cv=5).mean(),
        }
        mlflow.log_metrics({k: round(float(v), 4) for k, v in metrics.items()})

        # ---- log artifacts: plot + model with signature ----
        ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Died", "Survived"])
        plt.title(name)
        plt.savefig(HERE / "cm_tmp.png", dpi=110, bbox_inches="tight"); plt.close()
        mlflow.log_artifact(str(HERE / "cm_tmp.png"), artifact_path="plots")
        (HERE / "cm_tmp.png").unlink()

        signature = infer_signature(X_test, pipe.predict(X_test))
        mlflow.sklearn.log_model(pipe, name="model", signature=signature, input_example=X_test.head(3),
                                 serialization_format="cloudpickle")

        results.append((name, run.info.run_id, metrics["f1"], metrics["accuracy"]))
        print(f"{name:20s} accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} roc_auc={metrics['roc_auc']:.4f}")

# ---- pick the best run (by F1, tie-break accuracy) and register it ----
best_name, best_run_id, best_f1, best_acc = max(results, key=lambda r: (r[2], r[3]))
print(f"\nBest run: {best_name} (f1={best_f1:.4f}, accuracy={best_acc:.4f})")

mv = mlflow.register_model(f"runs:/{best_run_id}/model", REGISTERED_MODEL)
client = MlflowClient()
client.set_registered_model_alias(REGISTERED_MODEL, "champion", mv.version)
client.set_model_version_tag(REGISTERED_MODEL, mv.version, "validated_by", "quality_gate_f1")
print(f"Registered '{REGISTERED_MODEL}' version {mv.version} with alias @champion")
print("\nOpen the UI:  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001")
