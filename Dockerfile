FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

RUN pip install --no-cache-dir poetry==2.4.3

# Cache dependencies separately from application code.
COPY pyproject.toml poetry.lock ./
RUN poetry --no-cache install --only main,web --no-root --no-interaction --no-ansi

RUN useradd --create-home --uid 10001 app
COPY src/ ./src/
RUN poetry install --only-root --no-interaction --no-ansi

USER app
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "mlops.main:app", "--host", "0.0.0.0", "--port", "8000"]
