# Aletheia
A provenance-first knowledge engine that extracts, normalizes, and reconciles facts across documents, linking every conclusion to source evidence while distinguishing corroboration, contradiction, and contextual differences.

## Status
Phase 0 (backend foundation) — see `docs/PLAN.md` (requirements spec) and `docs/PROJECT_KNOWLEDGE_GRAPH.json` (engineering source of truth).

## Backend quickstart (Phase 0)
Requirements: Python 3.12, Docker + Docker Compose.

```powershell
# 1. Configure environment
Copy-Item .env.example .env

# 2. Start PostgreSQL + pgvector and the API
docker compose up --build

# 3. Health check (app vs dependency health are distinguished)
Invoke-RestMethod http://localhost:8000/health
```

Local run without Docker (PostgreSQL must be reachable via `DATABASE_URL`):

```powershell
py -V:3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt -r backend/requirements-dev.txt
$env:DATABASE_URL = "postgresql+psycopg://aletheia:aletheia@localhost:5432/aletheia"
uvicorn app.main:app --app-dir backend/app --port 8000
```

Run tests:

```powershell
py -V:3.12 -m pytest backend/tests -v
```

## Endpoints (Phase 0)
- `GET /health` — always 200 when the app is up; body includes nested `database: {configured, reachable, detail}`.

Full setup/architecture documentation lands in Phase 13 per `docs/PLAN.md`.
