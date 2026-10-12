"""Load the @champion model from the MLflow Model Registry and use it (what a deployment would do)."""
import mlflow
import pandas as pd

from pipeline import HERE

mlflow.set_tracking_uri(f"sqlite:///{(HERE / 'mlflow.db').as_posix()}")
model = mlflow.sklearn.load_model("models:/titanic-survival-model@champion")

people = pd.DataFrame([
    {"Pclass": 1, "Sex": "female", "Age": 28, "SibSp": 0, "Parch": 0, "Fare": 100.0, "Embarked": "C"},
    {"Pclass": 3, "Sex": "male", "Age": 30, "SibSp": 0, "Parch": 0, "Fare": 7.5, "Embarked": "S"},
])
people["prediction"] = ["SURVIVED" if p else "NOT SURVIVED" for p in model.predict(people)]
print(people.to_string(index=False))
