"use client";

import { useState } from "react";
import { ComparabilityPill } from "@/components/Pill";
import { pageImageUrl, type Evidence, type Fact } from "@/lib/api";

function normalized(fact: Fact) {
  if (fact.normalized_number === null && !fact.normalized_unit) return null;
  const n =
    fact.normalized_number === null
      ? ""
      : Number.isInteger(fact.normalized_number)
        ? String(fact.normalized_number)
        : String(+fact.normalized_number.toPrecision(10));
  return [n, fact.normalized_unit].filter(Boolean).join(" ");
}

function pageLabel(ev: Evidence) {
  const printed = ev.source_page_number ? ` (printed p. ${ev.source_page_number})` : "";
  return `PDF p. ${ev.pdf_page_number + 1}${printed}`;
}

export function FactCard({
  fact,
  docName,
  evidence,
}: {
  fact: Fact;
  docName: string;
  evidence: Evidence[];
}) {
  const [open, setOpen] = useState(false);
  const norm = normalized(fact);
  const mine = evidence.filter((e) => fact.evidence_ids.includes(e.id));

  return (
    <div className="surface mb-3 border-l-2 border-l-[#3a3a3a] p-4">
      <div className="label">{fact.subject}</div>
      <div className="mt-0.5 text-[17px] font-semibold">{fact.predicate}</div>
      <div className="my-1 text-[25px] font-bold">{fact.value_text}</div>

      {norm && (
        <>
          <div className="label mt-2">Normalized</div>
          <div className="text-sm text-accent">{norm}</div>
        </>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-x-2 gap-y-2 text-xs text-muted">
        <b className="text-[#b5b5b5]">{fact.time_text ?? "time unknown"}</b>
        <span>· {(fact.extraction_confidence * 100).toFixed(0)}% confidence</span>
        {fact.scope_text && <span>· {fact.scope_text}</span>}
        {fact.geography && <span>· {fact.geography}</span>}
        <ComparabilityPill fact={fact} />
      </div>
      <div className="mt-1 text-xs text-muted">
        {docName}
        {mine[0] ? ` · ${pageLabel(mine[0])}` : ""}
      </div>

      <button
        onClick={() => setOpen((v) => !v)}
        className="mt-3 rounded border border-[#3a3128] px-3 py-1.5 text-xs font-semibold tracking-wider hover:border-accent hover:text-accent"
      >
        {open ? "HIDE" : "EVIDENCE / PROVENANCE"}
      </button>

      {open && (
        <div className="mt-3">
          <div className="label mb-2">Fact → evidence → document → PDF page</div>
          {fact.ambiguity_flags.length > 0 && (
            <div className="mb-2 text-xs text-muted">
              flags: {fact.ambiguity_flags.join(", ")}
            </div>
          )}
          {mine.length === 0 && (
            <div className="text-xs text-muted">No evidence resolved for this fact.</div>
          )}
          {mine.map((ev) => (
            <div key={ev.id} className="mb-4">
              <div className="mb-1 text-xs text-muted">
                {pageLabel(ev)} · {ev.type} · {ev.extraction_method}
              </div>
              <pre className="surface max-h-56 overflow-auto p-3 font-mono text-xs whitespace-pre-wrap text-[#b5b5b5]">
                {ev.text.slice(0, 2000)}
              </pre>
              {ev.bbox && (
                /* The box only means anything drawn on the page it came from. */
                <img
                  src={pageImageUrl(fact.document_id, ev.pdf_page_number, ev.id)}
                  alt={`${pageLabel(ev)} with the evidence highlighted`}
                  className="surface mt-2 w-full"
                  loading="lazy"
                />
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
