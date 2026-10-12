# Stage 4 – CI: automated training, tests and ML quality gate (GitHub Actions)

**Concept:** every push automatically trains the model, **blocks** it if quality is too low (quality gate), and runs unit tests. (Practical 2 + 3)

| File | Purpose |
|---|---|
| `train_model.py` | Trains the model → `titanic_model.pkl`, `metrics.json`, `test_predictions.csv` |
| `quality_gate.py` | Fails the job (exit 1) if accuracy < 0.75 or F1 < 0.65 (`MIN_ACCURACY`/`MIN_F1` env vars) |
| `test_ml_pipeline.py` | 5 unit tests (dataset columns, outputs exist, metrics sane, binary predictions, missing values handled) |
| workflow | `../.github/workflows/stage4-ci-quality-gate.yml` |

## Run locally
```
pip install -r requirements.txt
python train_model.py
python quality_gate.py
python -m unittest discover -v
```
## Run on GitHub
Actions → **Stage 4 - ML CI + Quality Gate** → *Run workflow* (or push a change inside this folder).

**Failure demo without editing code:** *Run workflow* with `min_accuracy = 0.95` → *Check ML quality gate* turns red, tests are skipped. Run it again with `0.75` → green.
