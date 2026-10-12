"""Stage 1 - the ML pipeline: ingest -> preprocess -> train -> evaluate -> save."""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split

from pipeline import FEATURES, HERE, TARGET, build_pipeline, load_data

print("[1/5] Ingest data")
df = load_data()
X, y = df[FEATURES], df[TARGET]

print("[2/5] Split train/test (80/20, stratified)")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("[3/5] Train pipeline (preprocessing + RandomForest)")
pipe = build_pipeline()
pipe.fit(X_train, y_train)

print("[4/5] Evaluate")
pred = pipe.predict(X_test)
proba = pipe.predict_proba(X_test)[:, 1]
metrics = {
    "accuracy": round(accuracy_score(y_test, pred), 4),
    "precision": round(precision_score(y_test, pred), 4),
    "recall": round(recall_score(y_test, pred), 4),
    "f1": round(f1_score(y_test, pred), 4),
    "roc_auc": round(roc_auc_score(y_test, proba), 4),
    "cv_accuracy_mean": round(cross_val_score(build_pipeline(), X_train, y_train, cv=5).mean(), 4),
}
print(json.dumps(metrics, indent=2))

print("[5/5] Save model, metrics, confusion matrix")
joblib.dump(pipe, HERE / "titanic_model.pkl")
(HERE / "metrics.json").write_text(json.dumps(metrics, indent=2))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Died", "Survived"])
plt.title("Confusion matrix (test set)")
plt.savefig(HERE / "confusion_matrix.png", dpi=120, bbox_inches="tight")
print("Done: titanic_model.pkl, metrics.json, confusion_matrix.png")
