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

Fact extraction: `POST /documents/{id}/facts` runs evidence-scoped Groq extraction over one page-chunk at a time (never whole documents, never at upload time) and persists validated facts with evidence links; returns per-chunk counts plus rejection/error details, or 503 when no API key is configured. `GET /documents/{id}/facts` lists persisted facts. Every fact carries subject/predicate/value/unit/time/scope/context, deterministic numeric/date parsing where unambiguous, extraction confidence with ambiguity flags, and at least one evidence ID — unsupported content is rejected, never stored. Configure with `GROQ_API_KEY` and `GROQ_MODEL` (default `openai/gpt-oss-120b`).

Fact normalization: `POST /documents/{id}/normalize` deterministically converts persisted facts into canonical comparison form without touching raw fields, evidence links, confidence, or status — e.g. `₹81,415 Mn` → `8141.5 INR crore`, `1,429K tonnes` → `1.429 million tonnes`, `6.5%` → `0.065 fraction`. No LLM is involved; unknown or ambiguous units yield explicit `norm:` flags instead of invented values, and fiscal periods stay dateless.

Relationship reasoning: `POST /documents/{id}/relationships` discovers cross-document candidate pairs (MiniLM embeddings for ranking only, lexical fallback when unavailable, top-k bounded — never all-pairs) and classifies them deterministically into `CORROBORATES` / `CONTRADICTS` / `CONTEXTUAL_DIFFERENCE` / `RELATED` (`UNRELATED` is counted, not stored) with explanations, confidence, and `RESOLVED` / `NEEDS_REVIEW` status. Groq is consulted only for genuinely ambiguous pairs and never performs arithmetic; without a key those pairs persist as low-confidence review items. `GET /documents/{id}/relationships` lists them (type/min-confidence filters); `GET /relationships/{id}` returns both facts with their evidence. Embeddings run CPU-only (torch `2.14.0+cpu`, no CUDA stack); the Docker image pre-warms MiniLM weights at build time. Tune with `MATCHING_TOP_K`, `MATCHING_SIMILARITY_FLOOR`, and `MATCHING_NUMERIC_TOLERANCE`.

Knowledge-layer search: `POST /search` with `{query, document_id?, limit?, min_score?}` retrieves persisted facts without any LLM call. Independent lexical and vector discovery (reusing the MiniLM embeddings and token conventions, pgvector ordering on PostgreSQL) merge into deterministic scores — `lexical`, `vector`, or `hybrid` mean — ordered by score then fact ID, bounded by `limit`. Each hit returns the fact, score, match kind, source document, evidence, and persisted relationships. Blank queries are rejected with 400; the database is never mutated.

Full setup/architecture documentation lands in Phase 13 per `docs/PLAN.md`.
