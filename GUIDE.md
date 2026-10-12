# GUIDE – What you need to do, step by step

Project: **Titanic survival prediction, from model training to deployment** (MLflow → orchestration → ML pipeline → CI/CD).
Everything is in one folder `titanic-mlops/`; every stage has its own sub-folder. Do the stages **in order 1 → 9**.

## What was tested before you got this
| Stage | Tested here | Result |
|---|---|---|
| 1 ML pipeline | ran all scripts | accuracy 0.8156, F1 0.7179, ROC-AUC 0.8458 |
| 2 MLflow | ran 5 runs + registry + loaded `@champion`; started `mlflow ui` | best = gradient_boosting (F1 0.7317), registered as v1 `@champion`, UI answers HTTP 200 |
| 3 Orchestration | ran Prefect flow end-to-end, ran gate-failure case; ran Airflow 3.3.2 `dags test` end-to-end | all 7 tasks succeed; with `MIN_ACCURACY=0.95` task 5 fails and tasks 6–7 do not run |
| 4–9 code | trained, gated and ran all unit tests (stage 4: 5 tests, stages 6–9: 11 tests) | all pass |
| 8/9 container steps | ran the workflow's own "wait for /health" and "POST /predict" shell steps against the live API | pass |
| Not tested | `docker build/run` (no Docker in my sandbox) and the GitHub-hosted runs | they will run for the first time on your machine / GitHub – see Troubleshooting if anything is red |

---------------------------------------------------------------------

## PART 0 – One-time setup (10 min)
Install: **Python 3.11 or 3.12**, **Git**, **Docker Desktop** (start it), a **GitHub account**. Airflow additionally needs Linux/macOS/WSL2 (optional – Prefect covers orchestration on Windows).

```
cd titanic-mlops
python -m venv .venv
.venv\Scripts\activate                 # Linux/macOS: source .venv/bin/activate
pip install numpy pandas scikit-learn joblib matplotlib flask mlflow prefect
```
(Each stage also has its own `requirements.txt` if you prefer a clean env per stage.)

---------------------------------------------------------------------

## PART 1 – Local stages (no GitHub needed)

### Stage 1 – ML pipeline  (`cd stage1_ml_pipeline`)
```
python explore_data.py
python train.py
python predict_demo.py
```
Expected: 891 rows; survival women 74% vs men 19%; accuracy ≈ 0.8156. Files: `eda_survival.png`, `confusion_matrix.png`, `titanic_model.pkl`, `metrics.json`.
**Tell the TA:** preprocessing + model are one `Pipeline` object, so the same steps run in training and prediction.

### Stage 2 – MLflow  (`cd stage2_mlflow`)
```
python train_mlflow.py
python compare_runs.py
python load_registered_model.py
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```
Open http://127.0.0.1:5001 (takes ~15 s to start). Show: Experiments → `titanic-survival` → select all runs → **Compare**; open one run (params, metrics, artifacts); **Models** tab → `titanic-survival-model` → alias `champion`.
**Tell the TA:** every run is recorded and reproducible; the best model is registered and loaded by name, not by file path. Stop the UI with Ctrl+C.

### Stage 3 – Orchestration  (`cd stage3_orchestration`)
Prefect (any OS):
```
python flow.py
```
Shows 7 tasks completing in order. **Failure demo:** PowerShell `$env:MIN_ACCURACY="0.95"; python flow.py` → quality-gate task fails, later tasks are not run. Then `Remove-Item Env:MIN_ACCURACY`.
Optional dashboard: terminal 1 `prefect server start`; terminal 2 `$env:PREFECT_API_URL="http://127.0.0.1:4200/api"; python flow.py`; open http://127.0.0.1:4200.

Airflow (WSL/Linux/macOS, optional) – commands are in `stage3_orchestration/README.md` (copy 3 files + `data` into `$AIRFLOW_HOME/dags`, `airflow standalone`, trigger `titanic_training_pipeline`).
**Tell the TA:** an orchestrator adds ordering, retries, scheduling, logs and a UI around the same pipeline; the gate stops bad models from moving on.

---------------------------------------------------------------------

## PART 2 – Put the project on GitHub (5 min)
1. On github.com create a **new empty repository** (e.g. `titanic-mlops`, public or private, no README).
2. In the `titanic-mlops` folder:
```
git init
git add .
git commit -m "Titanic end-to-end MLOps project"
git branch -M main
git remote add origin https://github.com/<your-username>/titanic-mlops.git
git push -u origin main
```
`.github/` is a hidden folder – `git add .` includes it (browser drag-and-drop may not).
3. Repo → **Settings → Actions → General → Workflow permissions → Read and write permissions** (needed by Stage 9 to publish a package).
4. The first push starts workflows 4–9 automatically; open **Actions** and wait until all show green. (Later pushes only trigger the stage whose folder changed.)

---------------------------------------------------------------------

## PART 3 – GitHub stages (open **Actions**, pick the workflow on the left, **Run workflow**)

### Stage 4 – CI + quality gate  → "Stage 4 - ML CI + Quality Gate"
Green steps: Train → Check ML quality gate → Run automated ML tests.
**Failure demo:** Run workflow with `min_accuracy` = `0.95` → gate step red, tests skipped. Run again with `0.75` → green.
Screenshots: green run; red gate run with log "QUALITY GATE: FAIL".

### Stage 5 – Artifacts  → "Stage 5 - Preserve ML Artifacts"
Run → open the run → **Artifacts** → download `titanic-ml-artifacts` (model, metrics, test predictions).
Screenshots: Artifacts section; opened `metrics.json`.

### Stage 6 – Continuous Delivery  → "Stage 6 - Continuous Delivery (release candidate)"
Run → artifact `titanic-ml-release` (app, model, metrics, requirements, DEPLOYMENT.txt).
**Failure demo:** edit `stage6_continuous_delivery/test_app.py` on GitHub, in `test_high_survival_prediction` change `"SURVIVED"` → `"NOT SURVIVED"`, commit → *Run prediction application tests* red, release steps skipped. Restore → green.
Optional local test of the downloaded release: `pip install -r requirements.txt && python app.py`.

### Stage 7 – Docker (manual)  → "Stage 7 - Docker-ready release candidate"
Run the workflow, download `titanic-ml-release`, extract, then in that folder follow `stage7_docker/README.md` (Option A):
```
docker build -t titanic-survival-ml:1.0 .
docker run -d --name titanic-survival-ml-container -p 5000:5000 titanic-survival-ml:1.0
```
Test with the two PowerShell samples (SURVIVED / NOT SURVIVED), `docker logs`, port-mapping demo (run without `-p`), stop/remove/rerun.
(No time? Option B in that README builds straight from the repo folder.)

### Stage 8 – CI/CD with Docker  → "Stage 8 - Docker build and container tests"
Run → 12 green steps. Open *Build Docker image*, *Wait for application…* ("Application is ready."), *Test prediction endpoint* (`{"prediction":"SURVIVED",...}`).
**Failure demo:** in `.github/workflows/stage8-docker-cicd.yml` change `localhost:5000/health` → `localhost:5001/health`, commit → *Wait for application* red after 15 tries, logs + cleanup still run (`if: always()`). Change back → green.

### Stage 9 – Deployment  → "Stage 9 - Full CI/CD (train, test, containerize, deploy)"
Run → after the Docker tests the last two steps log in to GHCR and push `ghcr.io/<you>/titanic-survival-ml:latest` (+ commit-SHA tag). Check repo → **Packages**.
Deploy the published image on your PC (make the package Public first, see `stage9_deployment/README.md`):
```
$env:IMAGE="ghcr.io/<your-username-lowercase>/titanic-survival-ml:latest"
docker compose pull
docker compose up -d
Invoke-RestMethod http://localhost:5000/health
```
Stop: `docker compose down`. Pull-request runs test everything but **do not** publish.

---------------------------------------------------------------------

## PART 4 – Suggested demo order for the TAs (~20 min)
1. Show the repo tree and the stage table in `README.md` (1 min).
2. Stage 1 → run `train.py`, show confusion matrix (2 min).
3. Stage 2 → MLflow UI compare + registry (3 min).
4. Stage 3 → `python flow.py`, then the failure demo (3 min).
5. GitHub **Actions** page with all workflows green (1 min), then
   Stage 4 gate failure/pass (2 min) → Stage 5 artifact (1 min) → Stage 6 release artifact (1 min).
6. Stage 8 run log (Docker steps) (2 min) → Stage 9 package in GHCR + `docker compose up` + one prediction (3 min).
7. Close with the flow diagram: `data → pipeline → MLflow → orchestration → CI gate → artifact → API → Docker → registry → deployment`.

## Screenshot checklist
Stage 1: train output + confusion matrix · Stage 2: MLflow compare page, run detail, registered model · Stage 3: Prefect run output (success + failed gate) · Stage 4: green + red gate · Stage 5: artifacts list · Stage 6: green run, release artifact contents, failed API test run · Stage 7: `docker build` end, `docker images`, `docker ps` (5000->5000), browser health, SURVIVED & NOT SURVIVED results, `docker logs` · Stage 8: Build/Start/Health/Predict steps, red 5001 run, green final run · Stage 9: Packages page, `docker compose ps`, health response.

## Troubleshooting
| Problem | Fix |
|---|---|
| Workflow list is empty in Actions | `.github/workflows/` was not pushed – run `git add .github` and push |
| Stage 9 push step: *permission denied / 403* | Settings → Actions → General → Workflow permissions → **Read and write** |
| `docker compose pull` says *unauthorized/denied* | Make the package Public, or `docker login ghcr.io` with a token that has `read:packages`; username must be lowercase |
| `docker: command not found` / daemon error | Start Docker Desktop, reopen the terminal |
| Port 5000 busy | `-p 5001:5000` and use `localhost:5001` (Stage 7); or stop the other container |
| `FileNotFoundError: titanic_model.pkl` | Run `python train_model.py` in that stage folder first |
| `mlflow ui` page not opening | wait ~15 s after start; check port 5001 is free |
| Airflow won't install on Windows | use WSL2, or show Prefect for the orchestration stage |
| Quality gate fails unexpectedly | threshold is `MIN_ACCURACY`/`MIN_F1` in `quality_gate.py` (default 0.75 / 0.65) |
| Newer scikit-learn/pandas version complaints | stages train and serve in the same environment, so pickles stay compatible; keep Python 3.11+ |
