# mlops

FastAPI service for binary claim classification, with MLflow model loading,
async PostgreSQL access, health checks, containerized build and CI/CD.

## Stack

| Component | Version |
|---|---|
| Python | 3.14 |
| FastAPI | 0.141 |
| SQLAlchemy | 2.0 (async) |
| asyncpg | 0.31 |
| PostgreSQL | 17 |
| Poetry | 2.4.3 |
| Ruff | 0.16 |

## Architecture

```
src/mlops/
├── main.py              Application entry point, lifespan, request logging middleware
├── config.py            Pydantic settings
├── model.py             Registry alias resolution and model loading
├── core/
│   └── logging.py       Root logger configuration
├── db/
│   └── session.py       Async engine factory, get_db dependency
└── api/
    ├── healthz.py       Liveness probe
    ├── process.py       Text classification with the loaded model
    └── v1/
        ├── router.py    /api/v1 router
        ├── health.py    Readiness probe with PostgreSQL check
        └── version.py   Application version
```

Lifecycle:

- `lifespan` creates one `AsyncEngine` at startup and disposes it at shutdown.
- `async_sessionmaker` is stored in `app.state.session_factory`.
- `get_db` yields one `AsyncSession` per request and returns the connection to the pool on exit.
- Startup resolves the configured Registry alias, loads that exact model version once
  and stores it in `app.state.loaded_model`. Startup fails if the model cannot be loaded.
- `/process` calls the loaded model in a thread pool and returns its Registry version.

Engine configuration: `pool_pre_ping=True`, connect timeout 3s. Database health check is bounded by a 3s `asyncio.timeout`.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/healthz` | Liveness. No database access. |
| GET | `/api/v1/version` | Application version. |
| GET | `/api/v1/health` | Readiness. Runs `SELECT version()` against PostgreSQL. |
| POST | `/process` | Classifies `text`, returning the prediction and loaded model version. |

`/api/v1/health` response:

```json
{
  "status": "ok",
  "components": [
    {
      "name": "postgresql",
      "status": "up",
      "version": "PostgreSQL 17.0 ...",
      "response_time_ms": 4.21
    }
  ],
  "app_version": "0.1.0"
}
```

Returns HTTP 200 with `"status": "degraded"` and `"status": "down"` on the component when the database is unreachable.

`/process` accepts `{"text": "The Earth is flat."}`. Example response:

```json
{
  "prediction": "false",
  "model_name": "claim-veracity",
  "model_version": "1"
}
```

The version depends on the alias resolved at startup. Empty or whitespace-only text
returns 422; prediction failures return 500 without internal error details.
Swagger UI is available at `http://127.0.0.1:8001/docs` when using Compose.

## Environment variables

| Variable | Required | Default | Used by |
|---|---|---|---|
| `DB_HOST` | yes | — | application |
| `DB_PORT` | no | `5432` | application |
| `POSTGRES_DB` | yes | — | application, database |
| `POSTGRES_USER` | yes | — | application, database |
| `POSTGRES_PASSWORD` | yes | — | application, database |
| `APP_PORT` | no | `8001` | Docker Compose host port |
| `MLFLOW_DB_NAME` | yes | — | MLflow backend store |
| `MLFLOW_DB_USER` | yes | — | MLflow backend store |
| `MLFLOW_DB_PASSWORD` | yes | — | MLflow backend store |
| `S3_ACCESS_KEY` | yes | — | RustFS, MLflow server |
| `S3_SECRET_KEY` | yes | — | RustFS, MLflow server |
| `MLFLOW_BUCKET` | yes | — | MLflow artifact bucket |
| `MLFLOW_TRACKING_URI` | yes | — | application; Compose sets `http://mlflow:5000` |
| `MLFLOW_MODEL_NAME` | no | `claim-veracity` | application |
| `MLFLOW_MODEL_ALIAS` | no | `champion` | application |

Compose reads `.env` to configure its services. For a local application process,
Pydantic settings also read `.env` for MLflow configuration. Database connection
parameters are read directly from the process environment and must be exported.
Host-side training defaults to `http://127.0.0.1:5000`.

## Run with Docker Compose

```bash
cp .env.example .env
poetry install --with web,dev,ml
docker compose up --build -d --wait mlflow
```

For a fresh Registry, download MultiFC and follow the preparation and training
sections below. This creates `claim-veracity` versions and assigns `champion`.
If the model is already registered, proceed directly to application startup:

```bash
docker compose up --build -d --wait --wait-timeout 120 app
docker compose ps
```

The API is available at `http://127.0.0.1:8001`. On Windows PowerShell:

```powershell
$body = @{ text = "The Earth is flat." } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8001/process -Method Post `
    -ContentType "application/json" -Body $body
```

Inspect startup and inference logs:

```bash
docker compose logs --tail=100 app
```

Stop services while keeping database and MLflow volumes:

```bash
docker compose down
```

### Switch the working model

In MLflow UI, open **Model registry**, select `claim-veracity` and move `champion`
to another registered version. An already running application continues to use
the previous version. Call `/process` to verify that `model_version` is unchanged.

```bash
docker compose restart app
docker compose ps
```

Wait until app is healthy, then call `/process` again. `model_version` should now
match the new alias target; the prediction itself can remain the same. To roll back,
restore the alias to the previous version and restart the application again.

Changing the Registry alias alone does not replace an object already in memory.
In-process hot reload would need to coordinate active requests and model replacement
and may temporarily require memory for both versions. A controlled rollout of new
instances, followed by readiness checks and traffic switching, makes version changes
and rollback easier to observe.

## MLflow

`docker compose up` also starts an MLflow Tracking Server with separate stores:

| Service | Role | Address |
|---|---|---|
| `mlflow` | Tracking server and UI | `http://127.0.0.1:5000` |
| `mlflow-db` | Backend store (PostgreSQL 17): experiments, runs, params, metrics, dataset records | internal |
| `rustfs` | Artifact store (S3-compatible): plots, tables, models | console at `http://127.0.0.1:9001/rustfs/console/` |
| `bucket-init` | One-off job that creates the artifact bucket | exits with code 0 |

The server runs with `--artifacts-destination`, so it proxies artifacts: clients need only `MLFLOW_TRACKING_URI`, and the S3 keys stay inside the server container. The `mlflow` image is built from `docker/mlflow/Dockerfile`, which adds the PostgreSQL and S3 drivers to the official image.

`MLFLOW_SERVER_ALLOWED_HOSTS` allows `mlflow:5000` inside Compose and
`127.0.0.1:5000` / `localhost:5000` from the host. The internal service name must
be included for the application to access the Registry at startup.

To download artifacts from the host, set `MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false`. Otherwise the client asks for presigned URLs, which point at `rustfs:9000` and resolve only inside the compose network. Uploads and the UI are not affected.

### Dataset

[MultiFC](https://huggingface.co/datasets/pszemraj/multi_fc) (claim verification, 26 fact-checking sites). The files are not committed; download them into `data/multi_fc/`:

```bash
mkdir -p data/multi_fc
for split in train dev test; do
  curl -L -o data/multi_fc/$split.csv     https://huggingface.co/datasets/pszemraj/multi_fc/resolve/210f476c07dd5f486a9a51311e193a51c1ff8aec/$split.csv
done
```

### EDA and dataset tracking

```bash
poetry install --with ml
poetry run python experiments/eda.py
```

The script creates a run in the `multi_fc_eda` experiment and:

- registers the three splits with `mlflow.data.from_pandas` and links them to the run with `mlflow.log_input`;
- logs plots, tables and `summary.md` under the `eda/` artifact path;
- logs dataset statistics as metrics and the SHA-256 of each file as a param.

Dataset record fields:

- **source** — the Hugging Face URL pinned to a commit, so it always resolves to the same bytes.
- **digest** — MLflow's fingerprint of the DataFrame. For pandas it hashes the first 10 000 rows of the fully populated string and numeric columns, plus the row count and column names. It is not a checksum of the whole file, which is why the file SHA-256 is logged as well.
- **lineage** — the dataset-to-run link created by `log_input`. Runs that used a given dataset version can be found with `mlflow.search_runs(filter_string="dataset.name = 'multi_fc_train' AND dataset.digest = '<digest>'")`.

### Binary MultiFC preparation

```bash
poetry install --with web,dev,ml
poetry run python experiments/prepare_data.py
```

Only the original labels `true` and `false` are kept, as strings. All other labels
(including `mostly true`, `mostly false`, `half-true`, unknown verdicts and numeric
ratings) are excluded. This avoids guessing a common scale across fact-checking
sites. The model therefore covers the subset with explicit binary verdicts.

The script reads the original `train.csv` and `dev.csv` and:

- keeps only `claim` as `text` and `label`; verdict explanations and metadata are not features;
- removes empty texts and trims surrounding whitespace;
- matches duplicate texts using Unicode NFKC normalization, collapsed whitespace and case folding;
- removes all occurrences of texts with conflicting binary labels across both source splits;
- keeps one copy of each remaining text per source split;
- removes texts shared with dev from train, preserving dev for final evaluation;
- divides the remaining train rows into 80% train and 20% validation, stratified by label with seed 42.

Files are saved under `data/multi_fc/prepared/`:

| File | Contents |
|---|---|
| `train.csv` | Training data |
| `validation.csv` | Model and hyperparameter selection |
| `test.csv` | Cleaned original **dev**, reserved for final evaluation |
| `summary.json` | Label rule, seed, removal counts, class balance and source file SHA-256 hashes |

The original MultiFC `test.csv` has no labels and is not used for quality evaluation.
The prepared CSVs contain only `text` and `label`. Load labels explicitly as strings,
because pandas may otherwise interpret `true`/`false` as booleans:

```python
frame = pd.read_csv("data/multi_fc/prepared/train.csv", dtype={"text": "str", "label": "str"})
```

Prepared files stay outside Git under the existing `data/` ignore rule. Preparation
is local and does not require MLflow; the training runs will link the prepared datasets.

### Binary MultiFC training and Registry

Start MLflow and prepare the data before training. The application needs an existing
Registry model and alias, so start the tracking services first:

```bash
docker compose up --build -d mlflow
poetry run python experiments/prepare_data.py
poetry run python experiments/train.py
```

The script uses `MLFLOW_TRACKING_URI`, defaulting to `http://127.0.0.1:5000`, and creates
three runs in the `multi_fc_binary` experiment:

| Run | Model |
|---|---|
| `baseline_most_frequent` | Always predicts the most frequent training label |
| `tfidf_logistic_regression` | Word TF-IDF + balanced Logistic Regression, C=1 |
| `tfidf_linear_svc` | Word TF-IDF + balanced LinearSVC, C=1 |

Both learned models use word unigrams and bigrams, `min_df=2`, at most 50 000 features
and sublinear term frequency. There is no stop-word removal, so negations remain.
Balanced class weights account for the class imbalance. The vectorizer and classifier
are fitted only on train and saved together in a sklearn Pipeline. A ColumnTransformer
selects the `text` column, so prediction accepts a DataFrame directly.

Each run records classifier and TF-IDF parameters, a fixed seed, training time,
dataset inputs with split contexts, file SHA-256 hashes and the preparation report.
The dataset source is the absolute URI of the prepared CSV; the preparation report
links it back to the original train/dev files through their hashes. Training and
preparation scripts are also saved as artifacts.

Validation metrics include macro F1, accuracy, and precision/recall/F1 for both
classes. Artifacts include a confusion matrix, a classification report and the first
50 misclassified examples. Compare the runs in MLflow UI using `validation_macro_f1`.
Only the selected model is evaluated on prepared `test.csv` (the original dev).
Test results do not influence selection. Models are not refitted after selection.

Every saved model has an input example, signature and pinned package requirements.
The script reloads it through `mlflow.pyfunc`, checks this exact call and compares
its validation predictions with those of the fitted model:

```python
model.predict(pd.DataFrame({"text": ["Example claim"]}))
```

The output must contain one string `true` or `false` per row. No boolean or numeric
label conversion is performed. Both learned models are registered as versions of
`claim-veracity`. If the baseline wins on validation, it is registered too and can
become champion. Equal scores keep the first candidate in the table above.
The `champion` alias is assigned only after the selected Registry version has been
loaded and evaluated successfully. Existing versions remain available; rerunning
training creates new runs/versions and updates the alias.

Local results are saved to `data/multi_fc/training/comparison.csv` and `summary.json`.
The selected run also contains `comparison.json` and `training_summary.json`, with
the chosen candidate, run ID, Registry version and final metrics.

On Windows, keep temporary files on D if the system disk has little free space:

```powershell
New-Item -ItemType Directory -Force .tmp | Out-Null
$env:TEMP = Join-Path (Get-Location) ".tmp"
$env:TMP = $env:TEMP
$env:PYTHONUTF8 = "1"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
poetry run python experiments/train.py
```

Training defaults to proxied artifact downloads
(`MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false`), so the host does not need to reach
the internal RustFS address. The text-only classifiers learn dataset correlations;
they do not retrieve evidence or verify claims against external sources.

### Second dataset: Webis-Clickbait-17

[Webis Clickbait Corpus 2017](https://zenodo.org/records/5530410) (CC BY 4.0): 19 538 news posts from Twitter, each rated by 5 judges as clickbait or not. Download `clickbait17-train-170630.zip` from Zenodo and extract only the two JSONL files; the `media/` folder (910 MB of images) is not needed:

```bash
mkdir -p data/clickbait17
unzip -j clickbait17-train-170630.zip   clickbait17-validation-170630/instances.jsonl   clickbait17-validation-170630/truth.jsonl   -d data/clickbait17
```

```bash
poetry run python experiments/eda_clickbait.py
```

The run goes to the `clickbait17_eda` experiment. The dataset is registered as `clickbait17_train` with the Zenodo archive URL as source; the archive MD5 and the SHA-256 of both JSONL files are logged as params.

## Run locally

Requires Python 3.14, Poetry 2.4.3, a reachable PostgreSQL instance and an MLflow
Registry with the configured model and alias. Use database credentials matching
your PostgreSQL instance.

```bash
poetry install --with web,dev
```

```bash
export DB_HOST=127.0.0.1 DB_PORT=5432 POSTGRES_DB=mlops POSTGRES_USER=mlops POSTGRES_PASSWORD=mlops
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false
poetry run uvicorn mlops.main:app --reload --port 8000
```

On Windows PowerShell:

```powershell
$env:DB_HOST="127.0.0.1"; $env:DB_PORT="5432"; $env:POSTGRES_DB="mlops"; $env:POSTGRES_USER="mlops"; $env:POSTGRES_PASSWORD="mlops"
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
$env:MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD="false"
poetry run uvicorn mlops.main:app --reload --port 8000
```

## Build the Docker image

```bash
docker build -t mlops:local .
```

For startup with PostgreSQL and MLflow networking configured, use the Compose
instructions above. A standalone container also needs reachable database and
Registry addresses and the same model settings.

Image details: `python:3.14-slim` base, dependencies installed in a separate layer before application code, runtime group only (`--only main,web`), non-root user `app` (uid 10001), listens on port 8000.

## Tests

```bash
poetry run pytest
```

With coverage:

```bash
poetry run pytest --cov --cov-report=term-missing
```

Unit tests mock `AsyncSession` and the startup Registry loader. They require no
external services. Integration tests make HTTP requests to a separately running
application and are enabled explicitly:

```bash
docker compose up -d --wait --wait-timeout 120 app
RUN_DB_TESTS=1 RUN_MODEL_TESTS=1 TEST_BASE_URL=http://127.0.0.1:8001 poetry run pytest
```

On Windows PowerShell:

```powershell
$env:RUN_DB_TESTS="1"
$env:RUN_MODEL_TESTS="1"
$env:TEST_BASE_URL="http://127.0.0.1:8001"
poetry run pytest
```

The live tests check PostgreSQL connectivity and `/process` predictions with a
numeric Registry version. CI also sets `TEST_MODEL_VERSION` and
`TEST_MODEL_PREDICTION` to check the exact fixture version and output.

## Lint and format

```bash
poetry run ruff check .
```

```bash
poetry run ruff format .
```

Install git hooks:

```bash
poetry run pre-commit install
```

Hooks run `ruff check --fix`, `ruff format`, trailing whitespace, end-of-file, YAML syntax, merge conflict and large file checks.

## CI/CD

`.github/workflows/ci.yml` runs on every push and pull request. It:

- starts a PostgreSQL 17 service and installs the locked web/dev/ml dependencies;
- checks Ruff lint and formatting;
- starts an isolated MLflow server with SQLite backend and a separate filesystem artifact store under the runner's temporary directory;
- runs `tests/register_ci_model.py` to register a constant `false` model as version 1 of `claim-veracity` and assign `champion`;
- starts the application and waits for readiness;
- runs unit tests and both HTTP integration tests with coverage;
- uploads `coverage.xml` and MLflow/application logs, printing service logs on failure.

The small CI model checks Registry loading, the DataFrame/string contract and
HTTP inference. It is created in a fresh Registry for each job and does not change
the development Registry. MultiFC download and quality experiments are separate
from this API integration check. The fixture script is intended only for this
isolated CI server, not for registering a working model in the development environment.

`.github/workflows/cd.yml` runs after a successful CI run on `main`: builds the image from the tested commit and pushes it to `ghcr.io/<owner>/<repo>` with tags `latest`, `<version>`, `<version>-<short-sha>` and `sha-<short-sha>`.

## Dependency management

Dependencies are declared in `pyproject.toml` under `[dependency-groups]` (`web` for runtime, `dev` for tooling, `ml` for experiments) and pinned in `poetry.lock`.

```bash
poetry add --group web <package>
```

```bash
poetry lock
```

Both files must be committed together.
