import json
import unittest
from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "Titanic-Dataset.csv"


class TestMLPipeline(unittest.TestCase):
    def test_dataset_has_required_columns(self):
        df = pd.read_csv(DATA)
        for col in ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]:
            self.assertIn(col, df.columns)
        self.assertGreater(len(df), 800)

    def test_outputs_created(self):
        for f in ["titanic_model.pkl", "metrics.json", "test_predictions.csv"]:
            self.assertTrue((HERE / f).exists(), f"{f} missing - run train_model.py first")

    def test_metrics_reasonable(self):
        m = json.loads((HERE / "metrics.json").read_text())
        self.assertGreaterEqual(m["accuracy"], 0.75)
        self.assertLessEqual(m["accuracy"], 1.0)

    def test_model_predicts_binary(self):
        model = joblib.load(HERE / "titanic_model.pkl")
        row = pd.DataFrame([{"Pclass": 1, "Sex": "female", "Age": 30, "SibSp": 0,
                             "Parch": 0, "Fare": 80.0, "Embarked": "S"}])
        self.assertIn(int(model.predict(row)[0]), (0, 1))

    def test_model_handles_missing_age(self):
        model = joblib.load(HERE / "titanic_model.pkl")
        row = pd.DataFrame([{"Pclass": 3, "Sex": "male", "Age": None, "SibSp": 0,
                             "Parch": 0, "Fare": 8.0, "Embarked": None}])
        self.assertIn(int(model.predict(row)[0]), (0, 1))


if __name__ == "__main__":
    unittest.main()
