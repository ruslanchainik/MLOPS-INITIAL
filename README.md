# mlops

FastAPI service with async PostgreSQL access, health checks, containerized build and CI/CD.

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
├── core/
│   └── logging.py       Root logger configuration
├── db/
│   └── session.py       Async engine factory, get_db dependency
└── api/
    ├── healthz.py       Liveness probe
    └── v1/
        ├── router.py    /api/v1 router
        ├── health.py    Readiness probe with PostgreSQL check
        └── version.py   Application version
```

Lifecycle:

- `lifespan` creates one `AsyncEngine` at startup and disposes it at shutdown.
- `async_sessionmaker` is stored in `app.state.session_factory`.
- `get_db` yields one `AsyncSession` per request and returns the connection to the pool on exit.

Engine configuration: `pool_pre_ping=True`, connect timeout 3s. Database health check is bounded by a 3s `asyncio.timeout`.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/healthz` | Liveness. No database access. |
| GET | `/api/v1/version` | Application version. |
| GET | `/api/v1/health` | Readiness. Runs `SELECT version()` against PostgreSQL. |

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

The application reads these from the process environment. `.env` is consumed by Docker Compose, not loaded by the application itself.

## Run with Docker Compose

```bash
cp .env.example .env
docker compose up --build -d
```

Service is available at `http://127.0.0.1:8001`.

```bash
curl http://127.0.0.1:8001/api/v1/health
```

Stop:

```bash
docker compose down
```

Stop and delete database volume:

```bash
docker compose down -v
```

## MLflow

`docker compose up` also starts an MLflow Tracking Server with separate stores:

| Service | Role | Address |
|---|---|---|
| `mlflow` | Tracking server and UI | `http://127.0.0.1:5000` |
| `mlflow-db` | Backend store (PostgreSQL 17): experiments, runs, params, metrics, dataset records | internal |
| `rustfs` | Artifact store (S3-compatible): plots, tables, models | console at `http://127.0.0.1:9001/rustfs/console/` |
| `bucket-init` | One-off job that creates the artifact bucket | exits with code 0 |

The server runs with `--artifacts-destination`, so it proxies artifacts: clients need only `MLFLOW_TRACKING_URI`, and the S3 keys stay inside the server container. The `mlflow` image is built from `docker/mlflow/Dockerfile`, which adds the PostgreSQL and S3 drivers to the official image.

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

## Run locally

Requires Python 3.14, Poetry 2.4.3 and a reachable PostgreSQL instance.

```bash
poetry install --with web,dev
```

```bash
export DB_HOST=127.0.0.1 DB_PORT=5432 POSTGRES_DB=mlops POSTGRES_USER=mlops POSTGRES_PASSWORD=mlops
poetry run uvicorn mlops.main:app --reload --port 8000
```

On Windows PowerShell:

```powershell
$env:DB_HOST="127.0.0.1"; $env:DB_PORT="5432"; $env:POSTGRES_DB="mlops"; $env:POSTGRES_USER="mlops"; $env:POSTGRES_PASSWORD="mlops"
poetry run uvicorn mlops.main:app --reload --port 8000
```

## Build the Docker image

```bash
docker build -t mlops:local .
```

```bash
docker run --rm -p 8000:8000 \
  -e DB_HOST=host.docker.internal \
  -e POSTGRES_DB=mlops -e POSTGRES_USER=mlops -e POSTGRES_PASSWORD=mlops \
  mlops:local
```

Image details: `python:3.14-slim` base, dependencies installed in a separate layer before application code, runtime group only (`--only main,web`), non-root user `app` (uid 10001), listens on port 8000.

## Tests

```bash
poetry run pytest
```

With coverage:

```bash
poetry run pytest --cov --cov-report=term-missing
```

Unit tests mock `AsyncSession` and require no database. The integration test is skipped unless the service is running:

```bash
docker compose up -d
RUN_DB_TESTS=1 TEST_BASE_URL=http://127.0.0.1:8001 poetry run pytest
```

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

`.github/workflows/ci.yml` runs on every push and pull request: starts a PostgreSQL 17 service container, installs dependencies from `poetry.lock`, runs `ruff check`, `ruff format --check`, starts the application, runs the full test suite with coverage and uploads `coverage.xml`.

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
