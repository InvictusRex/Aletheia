# Aletheia
A provenance-first knowledge engine that extracts, normalizes, and reconciles facts across documents, linking every conclusion to source evidence while distinguishing corroboration, contradiction, and contextual differences.

## Status
Phase 1 (PDF ingestion + Canonical Evidence Layer) — see `docs/PLAN.md` (requirements spec) and `docs/PROJECT_KNOWLEDGE_GRAPH.json` (engineering source of truth).

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

## Endpoints (Phase 1)
- `GET /health` — always 200 when the app is up; body includes nested `database: {configured, reachable, detail}`.
- `POST /documents` — upload a PDF (multipart `file`, ≤50 MB); synchronously extracts pages/evidence and persists the bundle. Returns 201 with the document, page/evidence counts, and `ingestion_status` (`COMPLETED`/`PARTIAL`/`FAILED`). Non-PDF uploads are rejected with 400.
- `GET /documents/{id}` — inspect a document with its pages (quality verdicts) and evidence units (text, bbox, provenance).

Apply database migrations (PostgreSQL reachable via `DATABASE_URL`):

```powershell
alembic -c backend/alembic.ini upgrade head
```

Evidence model: every `EvidenceUnit` carries document ID, 0-based `pdf_page_number`, best-effort `source_page_number` (printed number, may be `None`), evidence type, text, bounding box, extraction method, and quality. Downstream phases consume this representation regardless of extractor.

Table evidence: pages with detected tables additionally yield `TABLE` units (tab/newline grid rendering plus `table_index`, dimensions, and verbatim headers in `meta`) and `TABLE_CELL` units (raw cell text with `row`/`col` plus verbatim `column_header`/`row_header` context). Native `TEXT` evidence is always preserved alongside tables. Disable with `TABLES_ENABLED=false`.

OCR fallback: pages whose native extraction quality verdict is `BAD` are additionally processed with local PaddleOCR (CPU, no cloud calls, no credentials). Successful OCR appends `OCR_TEXT` units with provider/trigger metadata; native evidence is always preserved and OCR failures never fail ingestion. The Docker image pre-warms model weights at build time; oneDNN/MKLDNN is intentionally disabled in the OCR engine because the current PaddlePaddle CPU oneDNN path cannot execute the OCR model kernels. Disable with `OCR_ENABLED=false`.

Fact extraction: `POST /documents/{id}/facts` runs evidence-scoped Gemini extraction over one page-chunk at a time (never whole documents, never at upload time) and persists validated facts with evidence links; returns per-chunk counts plus rejection/error details, or 503 when no API key is configured. `GET /documents/{id}/facts` lists persisted facts. Every fact carries subject/predicate/value/unit/time/scope/context, deterministic numeric/date parsing where unambiguous, extraction confidence with ambiguity flags, and at least one evidence ID — unsupported content is rejected, never stored. Configure with `GEMINI_API_KEY` and `GEMINI_MODEL` (default `gemini-3.5-flash`).

Fact normalization: `POST /documents/{id}/normalize` deterministically converts persisted facts into canonical comparison form without touching raw fields, evidence links, confidence, or status — e.g. `₹81,415 Mn` → `8141.5 INR crore`, `1,429K tonnes` → `1.429 million tonnes`, `6.5%` → `0.065 fraction`. No LLM is involved; unknown or ambiguous units yield explicit `norm:` flags instead of invented values, and fiscal periods stay dateless.

Full setup/architecture documentation lands in Phase 13 per `docs/PLAN.md`.
