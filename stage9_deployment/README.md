# Stage 9 – Deployment: the complete training-to-deployment pipeline

**Concept:** Stage 8 + **Continuous Deployment**. Only when *every* gate passed (quality gate → unit tests → container health → container prediction) does the pipeline publish the image to **GitHub Container Registry (GHCR)**. That image is the deployable unit; `docker compose` runs it anywhere.

```
push → train → quality gate → tests → docker build → run container → /health → /predict → PUSH TO GHCR → (deploy anywhere)
```
Workflow: `../.github/workflows/stage9-deploy.yml` (push to GHCR runs on `main`/manual runs, **skipped on pull requests**; uses the built-in `GITHUB_TOKEN`, no secrets to configure). Images: `ghcr.io/<owner>/titanic-survival-ml:<commit-sha>` and `:latest`.

## Run the pipeline
Actions → **Stage 9 - Full CI/CD** → *Run workflow*. After it is green open the repo → right side **Packages** → `titanic-survival-ml`.

## Deploy (run the published image)
First make the package readable: repo → Packages → `titanic-survival-ml` → *Package settings* → *Change visibility → Public* (or `docker login ghcr.io` with a personal access token that has `read:packages`).
```
$env:IMAGE="ghcr.io/<your-github-username-lowercase>/titanic-survival-ml:latest"
docker compose pull
docker compose up -d
docker compose ps
Invoke-RestMethod http://localhost:5000/health
```
(Linux/macOS: `IMAGE=ghcr.io/<user>/titanic-survival-ml:latest docker compose up -d`.)

## Deploy a locally built image instead
```
pip install -r requirements.txt && python train_model.py
docker compose up -d --build
```
Stop: `docker compose down`

The Dockerfile here also has a `HEALTHCHECK`; `docker ps` shows `(healthy)` after ~15 s.
