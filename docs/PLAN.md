# Fact Knowledge Layer — Project Plan

> **Status:** Canonical Project Plan / Single Source of Truth
>
> **Project:** Superjoin VIT 2026 Engineering Intern Take-Home Assignment
>
> **Core principle:** Build a provenance-first fact reconciliation system, not an LLM-powered PDF chatbot.

---

# 1. Project Objective

Build a **Fact Knowledge Layer** that can ingest one or more PDFs and automatically:

1. Extract meaningful numerical and semantic facts.
2. Ground every extracted fact in evidence from the source PDF.
3. Normalize facts expressed differently across documents.
4. Discover potentially equivalent or related facts across documents.
5. Identify whether facts:
   - corroborate each other,
   - genuinely/likely contradict each other,
   - differ because of context such as time, scope, units, or definitions,
   - or cannot be reliably reconciled.
6. Explain the reasoning behind relationships.
7. Handle extraction/reasoning failures explicitly.
8. Generalize to new PDFs without document-specific hardcoding.
9. Provide a simple API/UI for uploading PDFs and inspecting facts, evidence, and relationships.

The system should demonstrate that it can move from:

```
PDF
  ↓
Evidence
  ↓
Fact
  ↓
Normalization
  ↓
Candidate Matching
  ↓
Comparison
  ↓
Relationship
  ↓
Explanation
```

The important engineering problem is:

> **When are two facts actually the same, different, or contextually different?**

---

# 2. Assignment Success Criteria

The implementation must satisfy the following requirements.

## 2.1 Fact Extraction

Extract meaningful facts from PDFs, including:

- numerical facts,
- percentages,
- financial metrics,
- operational metrics,
- dates/time periods,
- quantities,
- semantic claims,
- entities,
- predicates/metrics,
- scope/context.

Facts must not exist independently of their source evidence.

---

## 2.2 Evidence Grounding

Every fact must point back to source evidence.

Evidence should ideally contain:

- document ID,
- PDF page number,
- source/printed page number when available,
- text,
- bounding box,
- evidence type,
- extraction method,
- extraction quality.

The UI should allow the user to inspect the source evidence for a fact.

Where possible, relevant text/table cells should be highlighted using bounding boxes.

---

## 2.3 Cross-Document Reasoning

The system must compare facts across documents.

Required relationship categories:

- `CORROBORATES`
- `CONTRADICTS`
- `CONTEXTUAL_DIFFERENCE`
- `RELATED`
- `UNRELATED`

The exact final taxonomy may evolve during implementation, but relationships must remain explicit and explainable.

---

## 2.4 Required Demonstration Cases

The final demo must demonstrate at least four cases.

### Case 1 — Corroboration

Two documents express the same fact differently.

Example:

```
₹81,415 Mn
≈ ₹8,141.5 Cr
≈ ₹8,142 Cr
```

The system should recognize these as the same underlying fact after normalization.

Another example:

```
740 Mn express parcels
740 Mn express parcel shipments
```

The system should recognize the semantic equivalence.

---

### Case 2 — Genuine / Likely Contradiction

Two facts refer to:

- the same entity,
- the same metric,
- the same time,
- compatible scope/context,

but contain materially different values.

The system should flag this as:

```
CONTRADICTS
```

or:

```
LIKELY_CONTRADICTION
```

depending on confidence.

The system must NOT blindly decide which source is correct.

---

### Case 3 — Apparent Contradiction Explained by Context

Two values appear different but are actually explained by context.

Possible contextual dimensions:

- time period,
- reporting vintage,
- scope,
- geography,
- unit,
- definition,
- population,
- methodology,
- estimate vs actual,
- forecast vs historical,
- preliminary vs revised value.

Example from the India macroeconomic dataset:

```
Economic Survey:
Real GDP growth = 6.4% for FY25

RBI:
Real GDP growth = 6.5% for 2024–25

IMF:
Real GDP growth = 6.5% for FY2024/25
```

The system should recognize that different estimates/vintages/context may explain the apparent difference rather than immediately declaring a contradiction.

---

### Case 4 — Extraction / Reasoning Failure

The system must demonstrate at least one situation where extraction or reasoning is uncertain or fails.

The system should:

- detect the uncertainty,
- preserve the evidence,
- avoid fabricating a fact,
- assign appropriate confidence,
- explain why extraction/reconciliation failed,
- optionally suggest how the pipeline could improve.

Failure handling is a feature, not an embarrassment.

---

# 3. Core Architectural Principle

The system must NOT become:

```
PDF → LLM → answer
```

Instead:

```
PDF
  ↓
Document Processing
  ↓
Canonical Evidence Layer
  ↓
LLM Fact Extraction
  ↓
Pydantic Validation
  ↓
Normalization
  ↓
Embeddings / Candidate Discovery
  ↓
Deterministic Comparison
  ↓
LLM Reasoning for Ambiguity
  ↓
Relationship Classification
  ↓
PostgreSQL
  ↓
API
  ↓
UI
```

The architecture must preserve provenance throughout the entire pipeline.

---

# 4. Technology Stack

## 4.1 Backend

- Python 3.12
- FastAPI
- Pydantic

---

## 4.2 PDF Processing

### Primary PDF engine

- PyMuPDF

Responsibilities:

- PDF loading,
- page iteration,
- native text extraction,
- text blocks,
- coordinates,
- metadata,
- page structure,
- image detection,
- extraction quality checks.

PyMuPDF is the primary PDF backbone.

---

### Table extraction

- pdfplumber

Use specifically for structured tables.

Tables must preserve:

- rows,
- columns,
- headers,
- cells,
- values,
- coordinates where possible.

Do NOT flatten financial tables into plain text if their structure can be preserved.

Example desired representation:

```json
{
  "row": "Revenue",
  "column": "FY24",
  "value": "81,415"
}
```

instead of:

```
Revenue FY24 FY23 81,415 ...
```

### OCR

- Local PaddleOCR (CPU)

PaddleOCR is an OCR fallback, not the default parser. It runs locally —
no cloud calls, no credentials — behind the OCR provider abstraction so
the engine can be replaced later.

Pipeline:

```
PDF
 ↓
PyMuPDF
 ↓
Extraction Quality Check
 ↓
 ┌───────────────┐
 │               │
GOOD            BAD
 │               │
 ↓               ↓
Continue       PaddleOCR
```

Do not OCR every page unnecessarily.

OCR should be used when native PDF extraction is absent, corrupted, or otherwise insufficient.

---

# 5. Canonical Evidence Layer

The Evidence Layer is the foundation of the entire system.

Every downstream component should consume the same canonical evidence representation regardless of whether the source came from:

- PyMuPDF,
- pdfplumber,
- PaddleOCR.

Do not independently send outputs from all three tools into downstream systems and attempt to merge duplicates later.

Instead:

```
PyMuPDF ────────┐
                │
pdfplumber ─────┼──→ Canonical Evidence Layer
                │
PaddleOCR ──────┘
```

## 5.1 EvidenceUnit

Initial conceptual model:

```python
class EvidenceUnit(BaseModel):
    id: str
    document_id: str
    page_number: int

    source_page_number: str | None

    type: str
    text: str

    bbox: tuple[float, float, float, float] | None

    extraction_method: str
    extraction_quality: float | None
```

The exact schema may evolve, but provenance must never be removed.

## 5.2 Evidence Types

Potential evidence types:

```
TEXT
TABLE
TABLE_ROW
TABLE_CELL
IMAGE
OCR_TEXT
HEADING
FOOTNOTE
CAPTION
```

The implementation should only introduce types that are actually useful.

---

# 6. Document Model

Each ingested PDF should receive a stable document ID.

Conceptual model:

```python
class Document(BaseModel):
    id: str
    filename: str
    title: str | None
    source: str | None

    page_count: int

    ingestion_status: str
    created_at: datetime
```

Important distinction:

**PDF page number vs source/printed page number**

Some supplied PDFs are curated excerpts.

Therefore:

- `page_number` = physical page index inside the uploaded PDF.
- `source_page_number` = page number printed/referenced by the original document when available.

Never assume they are identical.

---

# 7. Fact Model

A fact should conceptually represent:

```
Subject + Predicate + Value + Unit + Time + Scope + Context + Evidence
```

Initial model:

```python
class Fact(BaseModel):
    id: str

    subject: str
    canonical_subject: str

    predicate: str
    canonical_predicate: str

    value: ...
    normalized_value: ...

    unit: str | None
    normalized_unit: str | None

    time_context: ...

    scope: str | None
    context: ...

    evidence_ids: list[str]

    extraction_confidence: float
    normalization_confidence: float
```

The schema can evolve as implementation reveals real requirements.

---

# 8. Confidence Model

Avoid reducing all uncertainty to a single opaque number.

Where practical, track separate confidence dimensions:

```
Extraction confidence
        ↓
Evidence confidence
        ↓
Entity match confidence
        ↓
Normalization confidence
        ↓
Relationship confidence
```

This allows the UI to explain why a relationship is uncertain.

For example:

```
Relationship confidence: 0.71

Reason:
- Entity match: high
- Predicate match: high
- Numeric normalization: high
- Time context: uncertain
```

---

# 9. LLM Strategy

The LLM is used for semantic tasks, not as the entire reasoning engine.

Primary LLM:

- Gemini API — Free Tier

The Jio/consumer Gemini subscription does NOT provide developer API benefits, so the project will use the Gemini API free tier.

The implementation must remain provider-agnostic.

---

# 10. LLM Responsibilities

Gemini should be used for:

## 10.1 Fact Extraction

Input:

```
Evidence
```

Output:

```
Structured Fact
```

Example:

```
Evidence
   ↓
Gemini
   ↓
Structured JSON
   ↓
Pydantic validation
   ↓
Fact
```

## 10.2 Entity Normalization

Examples:

```
"Delhivery Ltd."
"Delhivery Limited"
"Delhivery"
```

→

```
canonical_entity = delhivery
```

## 10.3 Predicate Normalization

Examples:

```
"revenue from services"
"services revenue"
"revenue generated from services"
```

→

```
canonical_predicate = revenue_from_services
```

## 10.4 Ambiguous Relationship Reasoning

Gemini may be used when deterministic logic cannot confidently determine why two facts differ.

Example:

```
Fact A
+
Fact B
↓
Gemini
↓
Contextual difference
```

The LLM should receive the relevant structured facts and evidence, not entire documents.

---

# 11. What the LLM Must NOT Do

Do NOT use the LLM for deterministic operations when normal code is sufficient.

For example:

```
81,415 million INR
=
8,141.5 crore INR
```

should be handled by deterministic code.

Likewise:

```
6.4 ≠ 6.5
```

does not require an LLM.

The LLM should not be responsible for:

- basic arithmetic,
- unit conversion,
- date parsing where deterministic parsing works,
- exact numeric comparison,
- simple equality checks,
- provenance generation.

---

# 12. Structured LLM Output

LLM responses must be constrained to structured output.

Preferred pipeline:

```
Gemini
   ↓
Structured JSON
   ↓
Pydantic validation
   ↓
Validated Fact
```

Never allow arbitrary natural-language LLM output to become a database fact directly.

If the model produces malformed output:

- capture the failure,
- log the reason,
- preserve the source evidence,
- retry when appropriate,
- otherwise mark the extraction as failed/uncertain.

---

# 13. Gemini Provider Abstraction

Do not tightly couple the entire application to Gemini.

Use an abstraction such as:

```
LLMProvider
    │
    ├── GeminiProvider
    ├── OpenAIProvider       (future)
    └── LocalProvider        (future)
```

Conceptual interface:

```python
class LLMProvider(Protocol):

    async def extract_facts(
        self,
        evidence: list[EvidenceUnit]
    ) -> list[Fact]:
        ...

    async def normalize_entities(
        self,
        entities: list[str]
    ) -> ...:
        ...

    async def reason_about_relationship(
        self,
        fact_a: Fact,
        fact_b: Fact
    ) -> RelationshipReasoning:
        ...
```

The exact interface can evolve.

Environment configuration should look conceptually like:

```
LLM_PROVIDER=gemini
GEMINI_API_KEY=
```

Never commit API keys.

---

# 14. Gemini Usage Strategy

Because the project uses the Gemini API free tier, token efficiency matters.

Do NOT:

```
100-page PDF
    ↓
Gemini
```

Instead:

```
PDF
 ↓
Local extraction
 ↓
Evidence units
 ↓
Relevant evidence
 ↓
Gemini
```

Use Gemini selectively.

High-volume/simple tasks should use the most cost-efficient suitable Gemini model available.

More complex reasoning can use a stronger model when necessary and when the available API limits permit it.

The code should not hard-code assumptions about a particular model's availability beyond configuration.

---

# 15. Embeddings

Embeddings are for candidate discovery, not final truth determination.

Their purpose is:

```
"Which existing facts might be talking about the same thing?"
```

Example:

```
Fact A:
"India's real GDP growth was 6.5%"

Fact B:
"Real GDP expanded by 6.5 percent"

          ↓
      Embedding
          ↓
    Similarity Search
          ↓
    Candidate Pair
```

The candidate pair then moves to deterministic comparison/reasoning.

---

# 16. Embedding Model

Preferred initial option:

- Sentence Transformers

Reason:

- local,
- self-contained,
- no additional API dependency,
- suitable for candidate discovery,
- easy to replace later.

The exact embedding model should be selected during implementation based on:

- semantic quality,
- memory requirements,
- inference speed,
- dimensionality,
- compatibility with pgvector.

Do not spend excessive time optimizing embeddings before the core pipeline works.

---

# 17. Fact Matching Pipeline

Candidate matching should be hierarchical.

For a pair of facts:

```
1. Same / compatible entity?
        ↓
2. Same / compatible predicate?
        ↓
3. Compatible value type?
        ↓
4. Normalize units
        ↓
5. Compare numeric values
        ↓
6. Compare time context
        ↓
7. Compare scope
        ↓
8. Compare definitions/context
        ↓
9. Classify relationship
```

Embeddings should help identify candidate pairs before this stage.

They should NOT decide the final relationship by themselves.

---

# 18. Deterministic Normalization

Normalization should be performed locally whenever possible.

Examples:

**Currency**
```
₹81,415 Mn
→ 81,415 million INR
→ 8,141.5 crore INR
```

**Large numbers**
```
740 Mn
→ 740,000,000
```

**Tonnes**
```
1,429K tonnes
→ 1,429,000 tonnes
→ 1.429M tonnes
```

**Percentages**
```
6.5%
→ 0.065
```

The original representation must still be retained for provenance/display.

Never discard the original value.

---

# 19. Context Model

A numeric value without context is not necessarily a complete fact.

Relevant context dimensions include:

- fiscal year,
- calendar year,
- quarter,
- date,
- date range,
- estimate status,
- forecast/historical status,
- reporting vintage,
- geography,
- organizational scope,
- product/service category,
- methodology,
- unit,
- denominator,
- population,
- definition.

Example:

```
GDP growth = 6.5%
```

is incomplete without understanding:

```
6.5% of what?
For which period?
Which country?
Real or nominal?
Actual, estimate, or forecast?
Which reporting vintage?
```

The comparison engine must explicitly consider these dimensions.

---

# 20. Relationship Classification

The system should classify relationships using a combination of:

- deterministic rules,
- normalized values,
- metadata/context,
- embeddings,
- LLM reasoning only when ambiguity remains.

## 20.1 CORROBORATES

Use when:

- same entity,
- same predicate,
- compatible context,
- values are equivalent or sufficiently close after normalization,
- evidence comes from different sources.

Example:

```
81,415 Mn INR
≈
8,142 Cr INR
```

## 20.2 CONTRADICTS

Use when:

- same entity,
- same predicate,
- same/compatible time,
- same/compatible scope,
- same/compatible definition,
- values materially disagree.

Do not use this simply because:

```
A ≠ B
```

Context must first be checked.

## 20.3 CONTEXTUAL_DIFFERENCE

Use when values differ but the difference is explained by:

- different time period,
- different reporting vintage,
- different scope,
- different geography,
- different methodology,
- estimate vs actual,
- forecast vs historical,
- different definitions,
- other meaningful contextual distinctions.

## 20.4 RELATED

Use when two facts are semantically connected but cannot be considered equivalent or contradictory.

## 20.5 UNRELATED

Use when the candidate pair does not represent a meaningful relationship.

---

# 21. Relationship Reasoning Output

A relationship should not just contain:

```
CORROBORATES
```

It should contain an explanation.

Conceptual model:

```python
class Relationship(BaseModel):
    id: str

    fact_a_id: str
    fact_b_id: str

    relationship_type: str

    confidence: float

    explanation: str

    reasoning_metadata: dict
```

Example:

```
Relationship:
CORROBORATES

Explanation:
Both sources report Delhivery's FY24 revenue from
services. One reports ₹81,415 million while the other
reports approximately ₹8,142 crore. After unit
normalization, the values differ only by rounding.
```

---

# 22. Evidence Provenance

Every relationship should be traceable through:

```
Relationship
   ↓
Fact A
   ↓
Evidence A
```

```
Relationship
   ↓
Fact B
   ↓
Evidence B
```

The UI should make this chain visible.

The system should never produce a relationship that cannot be traced back to evidence.

---

# 23. Database

Use:

- PostgreSQL + pgvector

Core conceptual entities:

- documents
- pages
- evidence
- facts
- entities
- predicates
- relationships
- embeddings

PostgreSQL handles:

- relational data,
- provenance,
- facts,
- relationships,
- metadata.

pgvector handles:

- embedding storage,
- similarity search,
- candidate fact discovery.

---

# 24. Why PostgreSQL Instead of Neo4j

Do not introduce Neo4j initially.

The assignment explicitly emphasizes that a graph database/visualization alone is not the solution.

The core problems are:

- extraction,
- grounding,
- normalization,
- comparison,
- reconciliation,
- explanation.

PostgreSQL is sufficient for:

- Find facts about India
- Find facts about GDP
- Find similar facts
- Find conflicting facts
- Find evidence for fact X
- Find relationships involving fact X

Graph-like relationships can still be represented relationally.

A graph database can be considered later only if the prototype demonstrates a genuine need for it.

---

# 25. Redis / Queue / Distributed Processing

Do NOT introduce these initially:

- Redis
- Kafka
- RabbitMQ
- Celery
- Kubernetes

The assignment does not require distributed infrastructure.

Initial processing can use:

```
FastAPI
   ↓
Background task / simple worker
   ↓
PDF processing
```

If the prototype later requires a real queue, introduce one based on an actual bottleneck.

Do not add infrastructure merely for architectural appearance.

---

# 26. API

Use FastAPI.

Initial endpoints:

```
POST /documents
GET  /documents
GET  /documents/{id}

GET  /facts
GET  /facts/{id}

GET  /relationships
GET  /relationships/{id}

POST /search
```

Additional endpoints may be added if required by the UI.

FastAPI's automatic OpenAPI documentation should be retained.

---

# 27. Document Ingestion API

Conceptual workflow:

```
POST /documents
       ↓
Upload PDF
       ↓
Create document record
       ↓
Process PDF
       ↓
Extract evidence
       ↓
Extract facts
       ↓
Normalize facts
       ↓
Generate embeddings
       ↓
Find candidate matches
       ↓
Classify relationships
       ↓
Persist results
```

The API should expose processing status.

Possible statuses:

- `UPLOADED`
- `PROCESSING`
- `COMPLETED`
- `FAILED`
- `PARTIAL`

---

# 28. Frontend

Use:

- React
- Vite
- Tailwind CSS

Do not use Next.js unless there is a concrete reason.

SSR is not required.

The UI should prioritize:

- Explainability over aesthetics.

---

# 29. UI Components

Initial UI:

```
Dashboard
├── Upload
├── Documents
├── Facts
├── Relationships
└── Evidence Viewer
```

---

# 30. Upload View

The user should be able to:

- upload one PDF,
- upload multiple PDFs,
- see ingestion status,
- view document metadata,
- inspect processing errors.

---

# 31. Fact View

A fact should display:

- Subject
- Predicate
- Value
- Normalized Value
- Unit
- Time
- Scope
- Confidence
- Source

Example:

```
Delhivery
Revenue from Services
₹8,142 Cr
FY24
Confidence: High
```

Clicking the fact should reveal its evidence.

---

# 32. Relationship View

Display relationships clearly.

Example:

```
CORROBORATES

Revenue from Services
₹81,415 Mn

        ↕
   corroborates

Revenue from Services
₹8,142 Cr
```

Include:

- relationship type,
- confidence,
- explanation,
- both source documents,
- evidence.

---

# 33. Evidence Viewer

This is a high-value part of the UI.

Clicking:

```
Revenue = ₹8,142 Cr
```

should allow the user to inspect:

```
Source: Delhivery Earnings Presentation
PDF Page: 23
```

and see the relevant source text/table.

Where possible:

- render the PDF page,
- overlay/highlight the relevant bounding box,
- show extracted evidence.

This makes the provenance claim visually obvious.

---

# 34. Dataset Strategy

The supplied starter dataset contains two thematic datasets.

## Dataset A — Delhivery

Documents include:

- Prospectus excerpt,
- FY24 Annual Report excerpt,
- Q4 FY24 Earnings Presentation.

Useful overlapping facts include:

```
Revenue from services
₹81,415 Mn
≈ ₹8,142 Cr

EBITDA
₹1,266 Mn
≈ ₹127 Cr

Express parcels
740 Mn

PTL freight
1,429K tonnes
≈ 1.4M tonnes
```

These provide strong corroboration examples.

## Dataset B — India Macroeconomy

Documents include:

- Economic Survey 2024–25,
- RBI Annual Report 2024–25,
- IMF India 2025 Article IV.

Useful GDP-growth example:

```
Economic Survey:
6.4% real GDP growth

RBI:
6.5% real GDP growth

IMF:
6.5% real GDP growth
```

This is useful for demonstrating contextual differences / reporting-vintage differences rather than blindly declaring contradiction.

---

# 35. Page Numbering

The supplied PDFs are curated excerpts.

Some documents contain discontinuous source/printed page numbering.

Therefore the ingestion system must distinguish:

**PDF page number**

from:

**Original/source page number**

Both should be retained where possible.

---

# 36. Generalization Requirement

The system must work on PDFs that were not part of the original dataset.

Do NOT hard-code:

- Delhivery
- GDP
- Revenue
- FY24
- specific filenames
- specific page numbers
- known fact values
- known relationships

Document-specific examples may be used in:

- tests,
- demo fixtures,
- evaluation,
- README.

But the actual extraction/reasoning pipeline must remain generic.

---

# 37. No Hard-Coded Schemas Per Document

The system should NOT contain logic such as:

```python
if document == "delhivery":
    extract_revenue()
```

or:

```python
if page == 23:
    ...
```

or:

```python
if text contains "740 Mn":
    create_known_fact()
```

The schema should be general enough to represent facts discovered from arbitrary documents.

---

# 38. Dynamic Schema Philosophy

The fact schema should be stable at the structural level but flexible in content.

For example:

- Subject
- Predicate
- Value
- Unit
- Time
- Scope
- Context
- Evidence
- Confidence

should remain stable.

But predicates should not be hard-coded to only:

- revenue
- GDP
- EBITDA

The LLM and normalization layer should discover semantic predicates.

---

# 39. Extraction Quality Routing

Each page should receive an extraction quality assessment.

Potential signals:

- extracted character count,
- word count,
- percentage of suspicious characters,
- text density,
- presence of images,
- successful text extraction,
- table extraction quality.

If native extraction is sufficient:

```
Use native extraction
```

If not:

```
Route page to PaddleOCR
```

The routing threshold should be configurable.

---

# 40. Table Handling Strategy

Tables are particularly important because many numerical facts live inside them.

Pipeline:

```
PDF
 ↓
PyMuPDF
 ↓
Candidate table detection
 ↓
pdfplumber
 ↓
Structured table
 ↓
Evidence Units
 ↓
Fact Extraction
```

Table evidence must preserve enough structure to determine:

- which row
- which column
- which period
- which value

For example:

```
Revenue | FY24 | FY23
         ↓
81,415 | ...
```

The fact extractor must not accidentally associate a value with the wrong year.

---

# 41. Fact Extraction Strategy

The extraction prompt should instruct Gemini to:

- Extract only facts supported by supplied evidence.
- Never invent missing values.
- Preserve the original wording/value.
- Identify subject.
- Identify predicate.
- Identify value.
- Identify unit.
- Identify time context.
- Identify scope.
- Capture relevant qualifiers.
- Return evidence IDs.
- Assign extraction confidence.
- Flag ambiguous/uncertain facts.

The prompt should be generic and not mention the supplied datasets.

---

# 42. Evidence-First Rule

The system must follow this invariant:

```
No evidence → no fact.
```

If Gemini claims something that cannot be traced to supplied evidence:

```
Reject / mark invalid
```

Do not silently store it as truth.

---

# 43. Normalization Strategy

Normalization should preserve both:

**Original representation**
```
₹81,415 Mn
```

and:

**Normalized representation**
```
81415000000 INR
```

or an appropriate canonical representation.

This enables:

- accurate comparison,
- transparent UI,
- faithful source display.

Never replace the original source value with the normalized value.

---

# 44. Numeric Comparison

Numeric comparison should be deterministic.

Conceptual process:

```
Original Value
      ↓
Parse
      ↓
Unit Conversion
      ↓
Canonical Numeric Value
      ↓
Tolerance Check
```

Tolerance may be necessary for:

- rounding,
- approximate values,
- percentage reporting,
- display precision.

Example:

```
8,141.5 Cr
vs
8,142 Cr
```

should be considered equivalent if the contextual evidence indicates ordinary rounding.

The tolerance logic must be explicit and documented.

---

# 45. Semantic Comparison

Semantic equivalence requires more than string equality.

Examples:

```
express parcels
```

and:

```
express parcel shipments
```

may refer to the same metric.

Candidate discovery can use embeddings.

Final semantic interpretation can use:

- canonical predicates,
- entity normalization,
- contextual metadata,
- LLM reasoning where necessary.

---

# 46. Time Reasoning

Time should be treated as a first-class dimension.

Examples:

```
FY2024
2024–25
Q4 FY24
calendar year 2024
```

may or may not represent the same period.

The system should normalize time expressions but retain the original wording.

Do not assume:

```
2024 = FY24
```

without considering document/entity context.

---

# 47. Scope Reasoning

Scope must be compared explicitly.

Potential differences:

```
consolidated company
vs
standalone company
```

or:

```
India
vs
global
```

or:

```
all products
vs
one business segment
```

A scope mismatch can explain a numerical difference.

---

# 48. Estimate / Actual Reasoning

The system should distinguish:

- actual
- estimated
- forecast
- projected
- preliminary
- revised

when the source provides those qualifiers.

Two different estimates of the same metric should not automatically be treated as contradictions.

---

# 49. Reporting Vintage

Reporting vintage should be represented where available.

Example:

```
Economic Survey:
First Advance Estimate

RBI:
Later assessment

IMF:
Independent forecast/assessment
```

Different vintages may explain different values for the same period.

This is an important demonstration of contextual reasoning.

---

# 50. Failure Handling

Failures should be explicit.

Possible failure types:

- `EXTRACTION_FAILED`
- `OCR_FAILED`
- `TABLE_PARSE_FAILED`
- `LLM_PARSE_FAILED`
- `INSUFFICIENT_EVIDENCE`
- `AMBIGUOUS_FACT`
- `AMBIGUOUS_RELATIONSHIP`
- `NORMALIZATION_FAILED`

Never fabricate a result to make the pipeline appear successful.

---

# 51. Observability

At minimum, retain processing metadata such as:

- document processing status,
- extraction method,
- LLM request status,
- parsing failures,
- retry count,
- processing duration,
- number of evidence units,
- number of extracted facts,
- number of relationships,
- failed/uncertain facts.

Detailed production-grade telemetry is not required.

---

# 52. Testing Strategy

Tests should exist at multiple levels.

## Unit Tests

Test:

- unit conversion,
- numeric normalization,
- date normalization,
- percentage parsing,
- tolerance logic,
- relationship classification,
- evidence validation.

## Integration Tests

Test:

```
PDF
 ↓
Evidence
 ↓
Facts
 ↓
Normalization
 ↓
Relationships
```

using representative PDFs.

## Regression Tests

Keep known examples from the supplied dataset as regression cases.

For example:

```
₹81,415 Mn
↔
₹8,142 Cr
```

should continue to be recognized as corroboration.

---

# 53. Evaluation Dataset

Create a small internal evaluation suite containing:

- known corroborations,
- known contradictions,
- contextual differences,
- extraction failures,
- ambiguous cases.

Each evaluation case should specify:

- Expected relationship
- Expected reasoning
- Required evidence

Do not hard-code these into production logic.

---

# 54. Repository Structure

Recommended structure:

```
fact-knowledge-layer/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── documents.py
│   │   │   ├── facts.py
│   │   │   └── relationships.py
│   │   │
│   │   ├── extraction/
│   │   │   ├── pymupdf.py
│   │   │   ├── tables.py
│   │   │   ├── paddle_ocr.py
│   │   │   ├── ocr_provider.py
│   │   │
│   │   ├── facts/
│   │   │   ├── extractor.py
│   │   │   ├── normalizer.py
│   │   │   └── validator.py
│   │   │
│   │   ├── reasoning/
│   │   │   ├── matcher.py
│   │   │   ├── comparator.py
│   │   │   └── classifier.py
│   │   │
│   │   ├── models/
│   │   │   ├── document.py
│   │   │   ├── evidence.py
│   │   │   ├── fact.py
│   │   │   └── relationship.py
│   │   │
│   │   ├── llm/
│   │   │   ├── provider.py
│   │   │   └── gemini.py
│   │   │
│   │   ├── embeddings/
│   │   │   └── service.py
│   │   │
│   │   ├── db/
│   │   │   ├── models.py
│   │   │   ├── session.py
│   │   │   └── repositories/
│   │   │
│   │   └── main.py
│   │
│   └── tests/
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── ...
│   └── ...
│
├── scripts/
│
├── data/
│   └── README.md
│
├── docker-compose.yml
├── README.md
├── PLAN.md
├── .env.example
├── .gitignore
└── ...
```

The exact folder structure may evolve if implementation experience suggests a cleaner organization.

---

# 55. Implementation Phases

The project should be implemented incrementally.

Do NOT attempt to build the complete system at once.

## Phase 0 — Repository & Environment

Tasks:

- initialize repository,
- create Python environment,
- create FastAPI skeleton,
- create frontend skeleton,
- create Docker Compose,
- configure PostgreSQL,
- configure pgvector,
- create .env.example,
- ensure secrets are ignored.

Definition of done:

- Backend starts
- Frontend starts
- Database starts
- Health endpoint works

## Phase 1 — PDF Ingestion

Implement:

- document upload,
- document storage,
- PyMuPDF extraction,
- page metadata,
- text blocks,
- bounding boxes,
- extraction quality.

Definition of done:

```
Upload PDF
 ↓
Pages extracted
 ↓
EvidenceUnits generated
 ↓
Evidence persisted
```

## Phase 2 — Table Extraction

Implement:

- candidate table detection,
- pdfplumber integration,
- structured table extraction,
- table evidence units,
- row/column/cell provenance.

Definition of done:

```
Financial table
 ↓
Structured representation
 ↓
Evidence
```

## Phase 3 — OCR Fallback

Implement:

- extraction quality checks,
- PaddleOCR integration,
- routing logic,
- OCR evidence,
- failure handling.

Definition of done:

- Good native page → native extraction
- Bad/scanned page → OCR fallback

## Phase 4 — Fact Schema

Implement Pydantic models for:

- Document,
- EvidenceUnit,
- Fact,
- Relationship,
- Context,
- confidence metadata.

Definition of done:

All downstream components can rely on stable structured models.

## Phase 5 — Gemini Fact Extraction

Implement:

- Gemini provider,
- structured extraction prompt,
- Pydantic validation,
- retries,
- malformed-output handling,
- evidence ID enforcement.

Definition of done:

```
Evidence
 ↓
Gemini
 ↓
Validated Facts
```

## Phase 6 — Normalization

Implement deterministic normalization for:

- numeric values,
- currencies,
- units,
- percentages,
- dates,
- time periods.

Implement semantic normalization for:

- entities,
- predicates.

Definition of done:

Equivalent representations become comparable without losing original values.

## Phase 7 — Embeddings

Implement:

- Sentence Transformers,
- fact embedding generation,
- pgvector storage,
- similarity search.

Definition of done:

Given a fact, the system can discover likely related facts.

## Phase 8 — Comparison Engine

Implement hierarchical comparison:

```
Entity
 ↓
Predicate
 ↓
Value
 ↓
Unit
 ↓
Time
 ↓
Scope
 ↓
Context
```

Definition of done:

The system can distinguish:

- equivalent,
- different,
- contextual,
- unrelated.

## Phase 9 — Relationship Reasoning

Implement:

- deterministic relationship classification,
- ambiguity detection,
- Gemini reasoning for ambiguous cases,
- relationship explanation,
- confidence.

Definition of done:

Each meaningful relationship has:

- type
- confidence
- explanation
- fact A
- fact B
- evidence A
- evidence B

## Phase 10 — API

Implement:

- document APIs,
- fact APIs,
- relationship APIs,
- search API,
- processing status.

Definition of done:

The complete backend pipeline can be accessed through REST APIs.

## Phase 11 — UI

Implement:

- document upload,
- document list,
- fact explorer,
- relationship explorer,
- evidence viewer,
- PDF page/evidence highlighting where feasible.

Definition of done:

A reviewer can understand the system without using the database or API manually.

## Phase 12 — Evaluation

Run the full supplied datasets.

Verify:

- corroboration,
- contradiction,
- contextual difference,
- extraction failure.

Record:

- successful examples,
- failure cases,
- false positives,
- false negatives,
- limitations.

## Phase 13 — Demo & README

Prepare:

- polished README,
- architecture diagram,
- setup instructions,
- screenshots,
- sample outputs,
- limitations,
- trade-offs,
- AI tooling disclosure,
- demo video.

Demo should be <= 3 minutes.

---

# 56. Demo Strategy

The demo should tell a clear story.

Recommended flow:

**0:00–0:20**

Show the application.

Explain:

> "This system turns PDF claims into grounded facts and determines whether facts across documents corroborate, contradict, or differ because of context."

**0:20–0:50**

Upload documents.

Show processing.

**0:50–1:20**

Show extracted fact.

Click it.

Show:

- normalized value,
- confidence,
- source document,
- PDF page,
- evidence.

**1:20–1:50**

Show corroboration.

Example:

```
₹81,415 Mn
↔
₹8,142 Cr
```

Explain unit normalization and rounding.

**1:50–2:20**

Show contextual difference.

Example:

```
GDP growth = 6.4%
GDP growth = 6.5%
```

Show why this is not automatically treated as contradiction.

**2:20–2:40**

Show a contradiction/failure example.

**2:40–3:00**

Show architecture and briefly explain:

- evidence-first,
- deterministic normalization,
- embeddings for candidate discovery,
- LLM only where semantic reasoning is needed.

---

# 57. README Requirements

README must include:

## Setup

How to:

- install dependencies,
- configure environment,
- start PostgreSQL,
- start backend,
- start frontend,
- optionally configure Gemini.

## Run

Provide exact commands.

The reviewer should be able to reproduce the project.

## Architecture

Explain:

```
PDF
 ↓
Evidence
 ↓
Facts
 ↓
Normalization
 ↓
Matching
 ↓
Reasoning
 ↓
Relationships
```

## Design Decisions

Explain why:

- PyMuPDF,
- pdfplumber,
- PaddleOCR fallback,
- Gemini,
- Sentence Transformers,
- PostgreSQL,
- pgvector.

## Trade-offs

Discuss:

- free-tier LLM limits,
- OCR cost,
- embedding model choice,
- deterministic vs LLM reasoning,
- PostgreSQL vs graph database,
- prototype vs production architecture.

## AI Tools

Explicitly document AI tools used during development and how they were used.

Do not hide AI assistance.

## Limitations

Be honest about:

- OCR failures,
- table parsing edge cases,
- semantic ambiguity,
- LLM extraction errors,
- context interpretation,
- free-tier rate limits,
- PDF formatting variability.

## Next Steps

Potential future improvements:

- stronger document layout models,
- better table understanding,
- more sophisticated temporal reasoning,
- human verification,
- provenance visualization,
- incremental ingestion,
- richer search,
- evaluation benchmarks,
- production queue architecture.

---

# 58. Security Rules

NEVER commit:

- API keys,
- AWS credentials,
- .env,
- access tokens,
- passwords,
- private credentials.

Use:

```
.env
.env.*
```

where appropriate.

Commit:

```
.env.example
```

with placeholder values only.

---

# 59. Engineering Principles

The following principles are non-negotiable.

**Principle 1 — Evidence First**

```
No evidence → no fact.
```

**Principle 2 — Preserve Original Data**

Never overwrite the source representation with normalized values.

**Principle 3 — Deterministic Where Possible**

Do not use an LLM for arithmetic or obvious comparisons.

**Principle 4 — LLM for Semantics**

Use Gemini where language understanding is actually required.

**Principle 5 — Embeddings for Discovery**

Embeddings find candidates.

They do not determine truth.

**Principle 6 — Context Before Contradiction**

Different values do not automatically mean contradictory facts.

Check:

- time
- scope
- unit
- definition
- methodology
- vintage
- estimate/actual

first.

**Principle 7 — Never Hallucinate**

If evidence is insufficient:

```
UNCERTAIN
```

is better than a fabricated fact.

**Principle 8 — Explain Every Relationship**

The user should be able to understand:

```
Why did the system decide these two facts are related?
```

**Principle 9 — Generalize**

No document-specific production logic.

**Principle 10 — Don't Overengineer**

Avoid infrastructure that does not solve a demonstrated problem.

---

# 60. Explicit Anti-Patterns

Do NOT build:

```
PDF → Gemini → answer
```

Do NOT build:

```
PDF → embeddings → nearest neighbor = truth
```

Do NOT build:

```
PDF → OCR everything
```

Do NOT build:

```
Neo4j → graph visualization → assignment complete
```

Do NOT build:

```
LangChain + LangGraph + Redis + Celery + Kafka + Kubernetes
```

without an actual requirement.

Do NOT hard-code the supplied examples into the extraction engine.

Do NOT allow unsupported LLM claims to become facts.

---

# 61. Architecture Decision Summary

| Problem | Solution |
|---------|----------|
| PDF text/layout | PyMuPDF |
| Tables | pdfplumber |
| OCR | Local PaddleOCR fallback |
| Evidence | Canonical Evidence Layer |
| Fact extraction | Gemini |
| Structured output | Pydantic |
| Numeric normalization | Deterministic Python |
| Time normalization | Deterministic Python |
| Entity normalization | Gemini + deterministic cleanup |
| Predicate normalization | Gemini |
| Candidate matching | Sentence Transformers + pgvector |
| Final comparison | Deterministic rules |
| Ambiguous reasoning | Gemini |
| Storage | PostgreSQL |
| Vector search | pgvector |
| API | FastAPI |
| Frontend | React + Vite + Tailwind |
| Infrastructure | Docker Compose |

---

# 62. Final System

The final system should conceptually look like:

```
                         ┌─────────────────────┐
                         │       React UI      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       FastAPI       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │      Document Processing     │
                    │                              │
                    │ PyMuPDF → native extraction │
                    │ pdfplumber → tables         │
                     │ PaddleOCR → OCR fallback    │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │       Evidence Layer         │
                    │                              │
                    │ page / block / table / cell │
                    │ text / bbox / provenance    │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │       Fact Extraction        │
                    │                              │
                    │        Gemini API            │
                    │       + Pydantic             │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │        Normalization         │
                    │                              │
                    │ units / numbers / dates      │
                    │ entities / predicates        │
                    │ scope / context              │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │      Candidate Discovery     │
                    │                              │
                    │ Sentence Transformers        │
                    │ + pgvector                   │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │       Comparison Engine      │
                    │                              │
                    │ entity → predicate → value  │
                    │ time → scope → context       │
                    └──────────────┬───────────────┘
                                   │
                         ┌─────────┴─────────┐
                         │                   │
                         ▼                   ▼
                 Deterministic          Ambiguous
                   reasoning             reasoning
                         │                   │
                         │                   ▼
                         │              Gemini API
                         │                   │
                         └─────────┬─────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │        Relationships         │
                    │                              │
                    │ corroborates                 │
                    │ contradicts                  │
                    │ contextual difference        │
                    │ related / unrelated          │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │       PostgreSQL + pgvector   │
                    └──────────────────────────────┘
```

---

# 63. Definition of Done

The project is considered complete when all of the following are true:

## Core Pipeline

- [ ] PDFs can be uploaded.
- [ ] PDF pages are processed.
- [ ] Native text is extracted with PyMuPDF.
- [ ] Tables are extracted with pdfplumber.
- [ ] OCR fallback exists for problematic pages.
- [ ] Evidence Units are created.
- [ ] Evidence contains provenance.
- [ ] Facts are extracted with Gemini.
- [ ] Facts are validated using Pydantic.
- [ ] Facts retain evidence IDs.
- [ ] Values are normalized.
- [ ] Entities/predicates are normalized.
- [ ] Embeddings are generated.
- [ ] Candidate fact pairs are discovered.
- [ ] Relationships are classified.
- [ ] Ambiguous relationships can use Gemini reasoning.
- [ ] Relationship explanations are stored.
- [ ] Results are persisted in PostgreSQL.

## UI

- [ ] PDF upload works.
- [ ] Documents can be inspected.
- [ ] Facts can be inspected.
- [ ] Relationships can be inspected.
- [ ] Evidence can be inspected.
- [ ] Source page is visible.
- [ ] Bounding-box highlighting works where feasible.

## Assignment Demonstration

- [ ] Corroboration example works.
- [ ] Contradiction example works.
- [ ] Contextual-difference example works.
- [ ] Extraction/reasoning failure is demonstrated.

## Generalization

- [ ] No production logic is hard-coded to the supplied PDFs.
- [ ] New PDFs can be uploaded.
- [ ] The same pipeline processes them.

## Documentation

- [ ] README exists.
- [ ] Setup instructions work.
- [ ] Architecture is documented.
- [ ] Design decisions are documented.
- [ ] Trade-offs are documented.
- [ ] AI tooling is disclosed.
- [ ] Limitations are documented.
- [ ] Next steps are documented.
- [ ] Demo video is <= 3 minutes.

## Security

- [ ] No credentials are committed.
- [ ] .env.example exists.
- [ ] .gitignore is configured correctly.

---

# 64. Development Rule

When making implementation decisions that are not explicitly covered here:

- Prefer the simplest solution that preserves the architecture.
- Preserve provenance.
- Prefer deterministic logic over LLM reasoning when deterministic logic is sufficient.
- Prefer local processing when it reduces API usage.
- Do not introduce a new infrastructure component without a demonstrated need.
- Do not change the core architecture merely because another technology is fashionable.
- If a change materially affects the architecture, update this PLAN.md first.

---

# 65. Final Product Philosophy

The final submission should feel like:

> A small, well-engineered provenance and fact-reconciliation system.

It should NOT feel like:

> A giant AI stack wrapped around a PDF parser.

The strongest technical story is:

```
Documents
    ↓
Evidence
    ↓
Grounded Facts
    ↓
Normalization
    ↓
Candidate Discovery
    ↓
Context-Aware Comparison
    ↓
Explainable Relationships
```

Every layer should have a clear responsibility.

The system should be understandable by a reviewer.

The system should fail honestly.

The system should generalize beyond the supplied examples.

And most importantly:

> Every conclusion should be traceable back to the source evidence that caused the system to make it.