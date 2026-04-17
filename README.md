# Diagnostician

Prototype implementation of a React/Vite, source-grounded diagnostic reasoning game.

## What is included

- FastAPI backend with typed Pydantic contracts.
- SQLAlchemy/Postgres persistence with Alembic migrations and pgvector support.
- Local-source ingestion for de-identified JSON, Markdown, optional PDF files, and MultiCaRe Parquet case shards.
- Backend-authoritative gameplay workflows for starting runs, turns, diagnosis submission, scoring, and case review.
- Ollama adapters for generation and embeddings, with deterministic fallbacks for development when Ollama is unavailable.
- React + Vite + TypeScript client shell for the diagnostic workstation UI.
- Tests for ingestion, reveal policy, validation, scoring, and API behavior.

## Local backend setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e .[dev,pdf,multicare]
docker compose up -d postgres
$env:DIAGNOSTICIAN_DATABASE_URL="postgresql+psycopg://diagnostician:diagnostician@localhost:5432/diagnostician"
.\.venv\Scripts\python -m alembic -c backend/alembic.ini upgrade head
.\.venv\Scripts\python -m diagnostician.ingestion.cli ingest cases/source
.\.venv\Scripts\python -m uvicorn diagnostician.api.main:app --reload
```

## MultiCaRe data

Download the text-only MultiCaRe case database from Hugging Face:

```powershell
hf download OpenMed/multicare-cases --repo-type dataset --include '*.parquet' --include 'README.md' --local-dir data/multicare-cases
```

The parser also accepts the original `mauro-nievoff/MultiCaRe_Dataset` `cases.parquet` file. Avoid downloading the whole multimodal repo unless image assets are needed; the full repo includes multi-GB image archives.

The ingestion CLI accepts the downloaded Parquet shard directly. MultiCaRe rows are imported as review-draft cases because the dataset has source narratives and demographics, but no verified final-diagnosis field.

```powershell
.\.venv\Scripts\python -m diagnostician.ingestion.cli ingest data/multicare-cases --limit 25
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

## React client

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. The client expects the backend at `http://127.0.0.1:8000` by default. To point it elsewhere, set `VITE_API_BASE_URL` before running the dev server.
