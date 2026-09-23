/**
 * Every call to the Aletheia backend. Mirrors frontend/api.py: each helper
 * returns [data, error] so a page never has to decide what a thrown error
 * means, and never invents data when the backend fails.
 */

export const API =
  process.env.NEXT_PUBLIC_ALETHEIA_API_URL?.replace(/\/$/, "") ??
  "http://localhost:8000";

export type Result<T> = [T, null] | [null, string];

export interface Doc {
  id: string;
  filename: string;
  title: string | null;
  page_count: number;
  ingestion_status: string;
  created_at: string;
}

export interface Evidence {
  id: string;
  document_id: string;
  pdf_page_number: number;
  source_page_number: string | null;
  type: string;
  text: string;
  bbox: [number, number, number, number] | null;
  extraction_method: string;
  extraction_quality: number | null;
}

export interface Fact {
  id: string;
  document_id: string;
  subject: string;
  canonical_subject: string | null;
  predicate: string;
  canonical_predicate: string | null;
  value_kind: string;
  value_text: string;
  value_number: number | null;
  unit: string | null;
  normalized_number: number | null;
  normalized_unit: string | null;
  time_text: string | null;
  time_kind: string;
  scope_text: string | null;
  estimate_status: string;
  geography: string | null;
  evidence_ids: string[];
  extraction_confidence: number;
  ambiguity_flags: string[];
  status: string;
  comparability: "COMPARABLE" | "PARTIAL" | "NOT_COMPARABLE";
  missing_for_comparison: string[];
}

export interface Relationship {
  id: string;
  fact_a_id: string;
  fact_b_id: string;
  relationship_type: string;
  confidence: number;
  explanation: string;
  reasoning_metadata: Record<string, unknown>;
  status: string;
}

export interface RelationshipDetail {
  relationship: Relationship;
  fact_a: Fact;
  fact_b: Fact;
  evidence_a: Evidence[];
  evidence_b: Evidence[];
}

async function call<T>(path: string, init?: RequestInit): Promise<Result<T>> {
  try {
    const res = await fetch(`${API}${path}`, { cache: "no-store", ...init });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        detail = (await res.json())?.detail ?? detail;
      } catch {
        /* body was not JSON; the status line is the best we have */
      }
      return [null, `HTTP ${res.status}: ${detail}`];
    }
    return [(await res.json()) as T, null];
  } catch (err) {
    return [null, `API unreachable at ${API} — ${(err as Error).message}`];
  }
}

export const health = () => call<Record<string, any>>("/health");
export const listDocuments = () => call<Doc[]>("/documents");
export const getBundle = (id: string) =>
  call<{ document: Doc; pages: unknown[]; evidence: Evidence[] }>(`/documents/${id}`);
export const listFacts = (id: string) => call<Fact[]>(`/documents/${id}/facts`);
export const listRelationships = (id: string) =>
  call<Relationship[]>(`/documents/${id}/relationships`);
export const getRelationship = (id: string) =>
  call<RelationshipDetail>(`/relationships/${id}`);

export const search = (query: string, documentId?: string, limit = 30) =>
  call<{ query: string; hits: { fact: Fact; score: number; match_kind: string; document: Doc | null; evidence: Evidence[]; relationships: Relationship[] }[]; total: number }>(
    "/search",
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query, limit, ...(documentId ? { document_id: documentId } : {}) }),
    },
  );

export const runNormalize = (id: string) =>
  call<Record<string, any>>(`/documents/${id}/normalize`, { method: "POST" });
export const runRelationships = (id: string, recompute = false) =>
  call<Record<string, any>>(
    `/documents/${id}/relationships${recompute ? "?recompute=true" : ""}`,
    { method: "POST" },
  );

export async function uploadPdf(file: File): Promise<Result<Record<string, any>>> {
  const body = new FormData();
  body.append("file", file);
  return call("/documents", { method: "POST", body });
}

/** Rendered page with one evidence box drawn on it. */
export const pageImageUrl = (documentId: string, page: number, evidenceId?: string) =>
  `${API}/documents/${documentId}/pages/${page}/image` +
  (evidenceId ? `?evidence_id=${encodeURIComponent(evidenceId)}` : "");
