# Diagnostician

Demo implementation of a React/Vite, source-grounded diagnostic reasoning game.

## What is included

- FastAPI backend with typed Pydantic contracts.
- SQLAlchemy/Postgres persistence with Alembic migrations and pgvector support.
- Local-source ingestion for de-identified JSON, Markdown, optional PDF files, and MultiCaRe Parquet case shards.
- Ten approved demo cases with provenance, reveal policy, spoiler-locked diagnoses, and teaching points.
- Backend-authoritative gameplay workflows for starting runs, turns, diagnosis submission, scoring, and case review.
- Ollama adapters for generation and embeddings, with deterministic fallbacks for development when Ollama is unavailable.
- React + Vite + TypeScript diagnostic workstation with case browsing, replay avoidance, evidence panels, differential tracking, and review.
- MVP gameplay actions for history, exam, labs, ECG, imaging, procedures, treatments, consults, observation, progressive hints, and final diagnosis review.
- Tests for ingestion, reveal policy, validation, scoring, and API behavior.

## Fast demo setup

The Docker Compose stack starts Postgres, migrates the database, seeds the approved demo cases from `cases/source`, starts the FastAPI backend, and serves the built frontend.

```powershell
docker compose up --build
```

Open `http://127.0.0.1:5173`. The frontend calls the backend at `http://127.0.0.1:8000`.

Ollama is optional for the demo. If Ollama is unavailable and `DIAGNOSTICIAN_REQUIRE_OLLAMA=false`, generation and embeddings use deterministic development fallbacks so the game remains playable.

## Local backend setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e .[dev,pdf,multicare]
docker compose up -d postgres
$env:DIAGNOSTICIAN_DATABASE_URL="postgresql+psycopg://diagnostician:diagnostician@localhost:5432/diagnostician"
.\.venv\Scripts\python -m alembic -c backend/alembic.ini upgrade head
.\.venv\Scripts\python -m diagnostician.ingestion.cli seed-demo --path cases/source
.\.venv\Scripts\python -m uvicorn diagnostician.api.main:app --reload
```

## Local model setup

The backend can generate per-run case stories and audit them with local Ollama models. On Windows, run the setup script to scan hardware, write `.env`, pull the selected Qwen3 generator, pull the Med42 medical checker, and start the backend Docker services:

```powershell
.\scripts\setup-backend.ps1
```

The scanner picks a Qwen3 model from available RAM/VRAM. A 16 GB RAM laptop with integrated graphics selects the non-thinking `qwen3:4b-instruct` tag, disables the local Med42 checker, and keeps Qwen warm briefly so play relies on Qwen plus deterministic spoiler validation. Systems with at least 24 GB RAM or 8 GB VRAM keep Med42 enabled. Override any generated setting in `.env` before starting the backend if you want a specific model.

## MultiCaRe data

Download the text-only MultiCaRe case database from Hugging Face:

```powershell
hf download OpenMed/multicare-cases --repo-type dataset --include '*.parquet' --include 'README.md' --local-dir data/multicare-cases
```

The parser also accepts the original `mauro-nievoff/MultiCaRe_Dataset` `cases.parquet` file. Avoid downloading the whole multimodal repo unless image assets are needed; the full repo includes multi-GB image archives.

The ingestion CLI resolves MultiCaRe data local-first: root `cases.parquet`, then `data/multicare-cases/cases.parquet`, then an optional Hugging Face download. Automatically parsed rows with a diagnosis, safe opening facts, provenance, and enough revealable findings are approved for play; rows that cannot be structured are skipped into the ingestion report.

```powershell
.\.venv\Scripts\python -m diagnostician.ingestion.cli pull-multicare
.\.venv\Scripts\python -m diagnostician.ingestion.cli ingest-multicare cases.parquet --batch-size 500 --resume --skip-embeddings
.\.venv\Scripts\python -m diagnostician.ingestion.cli backfill-embeddings --batch-size 250
```

Use `--limit` and `--offset` for small test batches. Embeddings can be backfilled later; they are not required for a parsed case to be playable.

Ollama is expected at `http://localhost:11434`. If you do not use `scripts/setup-backend.ps1`, install and pull the configured models separately:

```powershell
ollama pull qwen3:4b-instruct
ollama pull hf.co/tensorblock/Llama3-Med42-8B-GGUF # only when DIAGNOSTICIAN_MEDICAL_CHECK_ENABLED=true
ollama pull nomic-embed-text
```

Set `DIAGNOSTICIAN_REQUIRE_OLLAMA=true` when you want ingestion and generation to fail instead of using deterministic development fallbacks.

The lower-level ingestion command is still available for arbitrary sources:

```powershell
.\.venv\Scripts\python -m diagnostician.ingestion.cli ingest cases/source
```

## Run tests

```powershell
.\.venv\Scripts\python -m pytest
```

Run the opt-in Docker Compose smoke test when Docker is available:

```powershell
$env:DIAGNOSTICIAN_RUN_DOCKER_SMOKE="1"
.\.venv\Scripts\python -m pytest tests/test_smoke.py
```

## React client

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. The client expects the backend at `http://127.0.0.1:8000` by default. To point it elsewhere, set `VITE_API_BASE_URL` before running the dev server.
