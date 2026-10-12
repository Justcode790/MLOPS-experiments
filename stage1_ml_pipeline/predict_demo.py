"""Load the saved model and predict for a few example passengers."""
import joblib
import pandas as pd

from pipeline import HERE

model = joblib.load(HERE / "titanic_model.pkl")
people = pd.DataFrame([
    {"Pclass": 1, "Sex": "female", "Age": 28, "SibSp": 0, "Parch": 0, "Fare": 100.0, "Embarked": "C"},
    {"Pclass": 3, "Sex": "male", "Age": 30, "SibSp": 0, "Parch": 0, "Fare": 7.5, "Embarked": "S"},
    {"Pclass": 2, "Sex": "female", "Age": 8, "SibSp": 1, "Parch": 1, "Fare": 30.0, "Embarked": "S"},
])
people["prediction"] = ["SURVIVED" if p else "NOT SURVIVED" for p in model.predict(people)]
people["probability"] = model.predict_proba(people.drop(columns="prediction"))[:, 1].round(3)
print(people.to_string(index=False))
