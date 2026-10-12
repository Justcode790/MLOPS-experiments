# Stage 5 – Preserve ML artifacts (GitHub Actions)

**Concept:** the GitHub runner is temporary; generated files vanish when the job ends unless uploaded as an **artifact**. Source code lives in the repo, generated model files live in artifacts. (Practical 4)

Same code as Stage 4 plus one workflow step that uploads, **only after the gate and tests pass**:
`titanic_model.pkl`, `metrics.json`, `test_predictions.csv` → artifact **titanic-ml-artifacts** (kept 7 days, `if-no-files-found: error`).

Workflow: `../.github/workflows/stage5-preserve-artifacts.yml`

## Run
Actions → **Stage 5 - Preserve ML Artifacts** → *Run workflow* → open the run → **Artifacts** section → download `titanic-ml-artifacts` and open `metrics.json`.

Local equivalent: `pip install -r requirements.txt && python train_model.py && python quality_gate.py && python -m unittest discover -v`
