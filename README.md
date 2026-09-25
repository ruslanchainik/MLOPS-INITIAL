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

Dependencies are declared in `pyproject.toml` under `[dependency-groups]` (`web` for runtime, `dev` for tooling) and pinned in `poetry.lock`.

```bash
poetry add --group web <package>
```

```bash
poetry lock
```

Both files must be committed together.
