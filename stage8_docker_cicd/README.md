# Stage 8 – CI/CD with Docker: automatic image build and container tests (GitHub Actions)

**Concept:** passing ML tests is not enough – the *final container* must build, start and answer requests. GitHub Actions now does what you did by hand in Stage 7. (Practical 7)

Workflow `../.github/workflows/stage8-docker-cicd.yml`:
train → quality gate → unit tests → **docker build** (tag = commit SHA) → **docker run** → **wait for /health** (15 retries × 2 s) → **POST /predict** → logs (`if: always()`) → cleanup (`if: always()`).
No registry push and no deployment in this stage.

## Run
Actions → **Stage 8 - Docker build and container tests** → *Run workflow*. All 12 steps should be green.
Screenshot: *Build Docker image* log, *Wait for application…* ending in `Application is ready.`, and *Test prediction endpoint* showing `{"prediction":"SURVIVED",...}`.

## Controlled failure demo
Edit the workflow: in *Wait for application to become ready* change `localhost:5000/health` → `localhost:5001/health`, commit. Build ✓, container start ✓, **Wait ✗** (after 15 tries), logs + cleanup still run. Change back to 5000 → green.

## Local check (no Docker needed)
```
pip install -r requirements.txt && python train_model.py && python -m unittest discover -v
```
