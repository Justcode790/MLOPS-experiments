"""Flask prediction API for the Titanic survival model."""
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path(__file__).resolve().parent / "titanic_model.pkl"
FIELDS = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
_model = None


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("titanic_model.pkl not found. Run train_model.py first.")
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/")
def root():
    return jsonify({"service": "titanic-survival-prediction", "status": "ok"})


@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON request body is required"}), 400

    missing = [f for f in FIELDS if f not in data]
    if missing:
        return jsonify({"error": "Missing required fields", "missing_fields": missing}), 400

    try:
        sample = pd.DataFrame([{
            "Pclass": int(data["pclass"]),
            "Sex": str(data["sex"]).lower(),
            "Age": float(data["age"]),
            "SibSp": int(data["sibsp"]),
            "Parch": int(data["parch"]),
            "Fare": float(data["fare"]),
            "Embarked": str(data["embarked"]).upper(),
        }])
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid field value"}), 400

    model = load_model()
    code = int(model.predict(sample)[0])
    prob = float(model.predict_proba(sample)[0][1])
    return jsonify({
        "prediction": "SURVIVED" if code == 1 else "NOT SURVIVED",
        "prediction_code": code,
        "survival_probability": round(prob, 4),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
