import unittest

from app import app

HIGH = {"pclass": 1, "sex": "female", "age": 28, "sibsp": 0, "parch": 0, "fare": 100, "embarked": "C"}
LOW = {"pclass": 3, "sex": "male", "age": 30, "sibsp": 0, "parch": 0, "fare": 7.5, "embarked": "S"}


class TestPredictionApplication(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["status"], "ok")

    def test_health_route(self):
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_high_survival_prediction(self):
        r = self.client.post("/predict", json=HIGH)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["prediction"], "SURVIVED")

    def test_low_survival_prediction(self):
        r = self.client.post("/predict", json=LOW)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["prediction"], "NOT SURVIVED")

    def test_missing_field_validation(self):
        r = self.client.post("/predict", json={"pclass": 1, "sex": "male"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("missing_fields", r.get_json())

    def test_invalid_value(self):
        bad = dict(HIGH, age="abc")
        self.assertEqual(self.client.post("/predict", json=bad).status_code, 400)


if __name__ == "__main__":
    unittest.main()
