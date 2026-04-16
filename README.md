# Diagnostician

Prototype implementation of a Godot-first, source-grounded diagnostic reasoning game.

## What is included

- FastAPI backend with typed Pydantic contracts.
- SQLAlchemy/Postgres persistence with Alembic migrations and pgvector support.
- Local-source ingestion for de-identified JSON, Markdown, and optional PDF files.
- Backend-authoritative gameplay workflows for starting runs, turns, diagnosis submission, scoring, and case review.
- Ollama adapters for generation and embeddings, with deterministic fallbacks for development when Ollama is unavailable.
- Godot 4.x GDScript client shell for the diagnostic workstation UI.
- Tests for ingestion, reveal policy, validation, scoring, and API behavior.

## Local backend setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e .[dev,pdf]
docker compose up -d postgres
$env:DIAGNOSTICIAN_DATABASE_URL="postgresql+psycopg://diagnostician:diagnostician@localhost:5432/diagnostician"
.\.venv\Scripts\python -m alembic -c backend/alembic.ini upgrade head
.\.venv\Scripts\python -m diagnostician.ingestion.cli ingest cases/source
.\.venv\Scripts\python -m uvicorn diagnostician.api.main:app --reload
```

Ollama is expected at `http://localhost:11434`. Install and pull the configured models separately:

```powershell
ollama pull llama3.1
ollama pull nomic-embed-text
```

Set `DIAGNOSTICIAN_REQUIRE_OLLAMA=true` when you want ingestion and generation to fail instead of using deterministic development fallbacks.

## Run tests

```powershell
.\.venv\Scripts\python -m pytest
```

## Godot client

Open the `godot/` folder in Godot 4.x and run `scenes/main.tscn`. The client expects the backend at `http://127.0.0.1:8000`.
