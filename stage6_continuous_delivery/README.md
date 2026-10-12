# Stage 6 – Continuous Delivery: prediction API + release candidate (GitHub Actions)

**Concept:** CI = verified code + model. **CD** = additionally validate the application that serves the model and automatically produce a deployable **release candidate**. (Practical 5)

| File | Purpose |
|---|---|
| `app.py` | Flask API: `GET /`, `GET /health`, `POST /predict` |
| `test_app.py` | 6 API tests (health, SURVIVED sample, NOT SURVIVED sample, missing fields → 400, invalid value → 400) |
| workflow | `../.github/workflows/stage6-continuous-delivery.yml` |

Release artifact **titanic-ml-release** = `app.py`, `titanic_model.pkl`, `metrics.json`, `requirements.txt`, `DEPLOYMENT.txt` – only created if every earlier step passed.

## Run locally
```
pip install -r requirements.txt
python train_model.py && python quality_gate.py && python -m unittest discover -v
python app.py                      # API on http://127.0.0.1:5000
```
Test it (PowerShell):
```
$body = @{ pclass=1; sex="female"; age=28; sibsp=0; parch=0; fare=100; embarked="C" } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -ContentType "application/json" -Body $body
```
curl: `curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"pclass":3,"sex":"male","age":30,"sibsp":0,"parch":0,"fare":7.5,"embarked":"S"}'`

## Failure demo
In `test_app.py` change `"SURVIVED"` to `"NOT SURVIVED"` in `test_high_survival_prediction`, push → *Run prediction application tests* is red and **Prepare/Upload release candidate are skipped**. Revert → green and artifact returns.
