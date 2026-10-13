# Letterboxd Movie Recommender

A small full-stack movie recommender. Upload a Letterboxd `ratings.csv` export, map unrated external IDs through Letterboxd, retrieve candidates with a two-tower model, and rerank them with XGBoost.

> **Project status:** research prototype. The web interface and API are present, but the recommender requires a large catalogue artifact that is not currently included in this repository. Follow [Data and model files](#data-and-model-files) before starting the API.

## How it works

1. The React interface sends a Letterboxd ratings CSV to the FastAPI service.
2. The parser uses existing TMDB IDs where present and resolves missing IDs from Letterboxd pages. It caches successful lookups under `data/letterboxd/cache.json`.
3. User features are calculated from the ratings and movie feature catalogue.
4. The user tower scores unseen catalogue movies against precomputed movie embeddings.
5. The top candidates are reranked by the saved XGBoost ranker. The API returns up to 10 titles with TMDB links.

## Repository layout

```text
backend/              FastAPI upload and recommendation endpoint
frontend/             React 19 + Vite interface
src/parser/           Letterboxd CSV parsing and ID mapping
src/recommender/      Feature building and ranking pipeline
models/               Saved Keras, scaler, and XGBoost artifacts
notebooks/            Data exploration, feature engineering, and model work
data/                 Local/generated data; only small required files are tracked
```

## Requirements

- CPython 3.12 is the supported baseline (`.python-version`). Use 64-bit CPython 3.12 for local installs and Docker. Other Python versions and operating systems have not been validated.
- Node.js and npm
- Git LFS, to check out the tracked Parquet catalogue (see below)
- Internet access when uploaded CSV rows need Letterboxd ID lookup

Python versions and operating systems have not yet been validated in CI. TensorFlow wheel availability and behavior can vary by platform.

## Setup

Clone the repository with Git LFS installed and initialized so the catalogue is downloaded rather than left as an LFS pointer:

```bash
git lfs install
git clone <repository-url>
cd letterboxd-rec
```

Create a Python 3.12 environment and install the API/recommender dependencies. `requirements.txt` is the single runtime entry point; `requirements/runtime.txt` holds its pinned packages:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For notebook and feature-engineering work, install the additional tools on top of the runtime environment:

```bash
python -m pip install -r requirements/notebooks.txt
```

The root `requirements.txt` includes `requirements/runtime.txt`, so use it for the runtime install. The notebook file includes the runtime set and adds notebook packages. Keep Docker and local runtime installs on the same root requirements file. Pins are explicit; transitive versions are resolved by pip during installation.

Install frontend dependencies:

```bash
cd frontend
npm ci
cd ..
```

Start the API from the repository root:

```bash
uvicorn backend.main:app --reload
```

In another terminal, start the UI:

```bash
cd frontend
npm run dev
```

Open the local URL printed by Vite (normally `http://localhost:5173`) and upload the `ratings.csv` file from a Letterboxd data export. The API accepts CSV uploads up to 10 MB. It can also accept a CSV already enriched with TMDB IDs. If the UI needs to call a non-default API URL, set `VITE_API_BASE_URL` in `frontend/.env.local` before starting Vite.

## Deployment

The repository includes Dockerfiles for the API and static frontend, plus a Compose configuration for a single-host deployment. The API image includes the saved models and catalogue artifacts, so the missing-artifact requirement below must be resolved before building it.

For Compose, copy `.env.deploy.example` to `.env` and set the public API URL:

```dotenv
VITE_API_BASE_URL=https://your-api.example.com
FRONTEND_PORT=8080
```

Then build and start:

```bash
docker compose --env-file .env up --build -d
```

The frontend is served on port 8080 by default. Put the services behind HTTPS-capable hosting or a reverse proxy and point the public API hostname at the API container's port 8000. For a platform that deploys containers separately, build from `Dockerfile.api` and `frontend/Dockerfile`; set the build argument `VITE_API_BASE_URL` on the frontend. The URL is embedded into the static frontend at build time, so rebuild after changing it. The API does not enable CORS; serve the frontend and API through one origin or configure the reverse proxy to route API requests.

The API accepts CSV files up to 10 MB, writes each upload to a temporary file only while processing it, and removes it when the request finishes. It keeps Letterboxd ID lookup results in `LETTERBOXD_CACHE_PATH` (default `data/letterboxd/cache.json`); the container defaults this to `/var/lib/letterboxd-rec/cache.json`. Mount persistent storage there if the cache should survive container replacement. The API serializes model inference to one request at a time, so set container memory/CPU based on the TensorFlow runtime and expect concurrent uploads to queue.

## Data and model files

The inference code requires these files:

| Path | Purpose | Current availability |
| --- | --- | --- |
| `data/movie_feature_frame.parquet` | Movie catalogue features and display titles | Tracked with Git LFS |
| `data/norm_movie_embeddings_nn.parquet` | Precomputed movie tower embeddings | **Required, but ignored and not tracked** |
| `data/genre_columns.csv` | Genre feature column list | Tracked |
| `models/userNN.keras` | User tower | Tracked |
| `models/scaler_users.pkl`, `models/scaler_y.pkl` | Input and output scaling | Tracked |
| `models/xgboostranker_model.json` | Final reranker | Tracked |

The app will not start recommendation inference until `norm_movie_embeddings_nn.parquet` is supplied in `data/`. It is generated by the model/data pipeline in the notebooks, but there is currently no documented one-command artifact build or public download. Do not publish it or the other derived datasets until their upstream source terms and redistribution permissions are checked. The larger training and evaluation data files are intentionally not committed.

The API's ID cache is generated automatically. It may contain Letterboxd identifiers and should remain local; do not commit a personal export or cache.

## Development notes

- API entry point: `backend.main:app`; recommendation endpoint: `POST /recommend`.
- Frontend: React + Vite (not Next.js); run `npm run dev`, `npm run build`, or `npm run lint` from `frontend/`.
- The notebooks describe exploratory and training work; they are not a clean, end-to-end model training pipeline yet.
- `src/recommender/hybrid_test.py` and `recommend.py` are evaluation/experimentation scripts and require additional local data files.

## License

This repository's code is provided under the MIT License; third-party data and model artifacts may have separate terms. See [LICENSE](LICENSE) and the notices accompanying each upstream dataset or artifact.
