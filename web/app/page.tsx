"use client";

import { useCallback, useEffect, useState } from "react";
import {
  health, listDocuments, listFacts, listRelationships,
  runNormalize, runRelationships, uploadPdf, type Doc,
} from "@/lib/api";

type Stats = { facts: number; comparable: number; relationships: number };

export default function Workspace() {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [stats, setStats] = useState<Record<string, Stats>>({});
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [db, setDb] = useState<string>("checking");

  const refresh = useCallback(async () => {
    const [h] = await health();
    setDb(h?.database?.reachable ? "connected" : "unreachable");
    const [list, err] = await listDocuments();
    if (err) return setError(err);
    setError(null);
    setDocs(list ?? []);
    const next: Record<string, Stats> = {};
    await Promise.all(
      (list ?? []).map(async (d) => {
        const [facts] = await listFacts(d.id);
        const [rels] = await listRelationships(d.id);
        next[d.id] = {
          facts: facts?.length ?? 0,
          comparable: (facts ?? []).filter((f) => f.comparability === "COMPARABLE").length,
          relationships: rels?.length ?? 0,
        };
      }),
    );
    setStats(next);
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  const run = async (label: string, fn: () => Promise<unknown>) => {
    setBusy(label);
    await fn();
    await refresh();
    setBusy(null);
  };

  const onUpload = async (files: FileList | null) => {
    if (!files?.length) return;
    await run("Uploading", async () => {
      for (const file of Array.from(files)) {
        const [, err] = await uploadPdf(file);
        if (err) setError(err);
      }
    });
  };

  const total = Object.values(stats).reduce(
    (acc, s) => ({
      facts: acc.facts + s.facts,
      comparable: acc.comparable + s.comparable,
      relationships: acc.relationships + s.relationships,
    }),
    { facts: 0, comparable: 0, relationships: 0 },
  );

  return (
    <div>
      {error && <div className="surface mb-4 border-l-2 border-l-red p-3 text-sm">{error}</div>}

      <div className="mb-6 grid grid-cols-2 gap-3 md:grid-cols-4">
        {[
          ["Documents", docs.length],
          ["Facts", total.facts],
          ["Comparable", total.comparable],
          ["Relationships", total.relationships],
        ].map(([label, value]) => (
          <div key={label as string} className="surface p-4">
            <div className="label">{label as string}</div>
            <div className="mt-1 text-3xl font-bold">{value as number}</div>
          </div>
        ))}
      </div>

      <div className="surface mb-6 p-4">
        <div className="label mb-2">Upload PDFs</div>
        <input
          type="file"
          accept="application/pdf"
          multiple
          onChange={(e) => void onUpload(e.target.files)}
          className="block w-full text-sm file:mr-3 file:rounded file:border file:border-[#3a3128] file:bg-transparent file:px-3 file:py-1.5 file:text-xs file:font-semibold file:text-foreground hover:file:border-accent"
        />
        <div className="mt-2 text-xs text-muted">
          Database {db}. Ingestion extracts pages and evidence; facts are extracted on demand.
        </div>
      </div>

      <div className="label mb-2">Documents</div>
      {docs.map((doc) => {
        const s = stats[doc.id];
        return (
          <div key={doc.id} className="surface mb-2 flex flex-wrap items-center gap-4 p-3">
            <div className="min-w-[280px] flex-1">
              <div className="font-semibold break-words">{doc.filename}</div>
              <div className="text-xs text-muted">
                {doc.page_count} pages · {doc.ingestion_status}
              </div>
            </div>
            <div className="text-xs text-muted">
              {s ? `${s.facts} facts · ${s.comparable} comparable · ${s.relationships} relationships` : "…"}
            </div>
            <div className="flex gap-2">
              <button
                disabled={busy !== null}
                onClick={() => void run("Normalizing", () => runNormalize(doc.id))}
                className="rounded border border-[#3a3128] px-3 py-1.5 text-xs font-semibold tracking-wider hover:border-accent hover:text-accent disabled:opacity-40"
              >
                NORMALIZE
              </button>
              <button
                disabled={busy !== null || (s?.comparable ?? 0) === 0}
                title={(s?.comparable ?? 0) === 0 ? "Normalize first — relationships compare normalized values" : ""}
                onClick={() => void run("Matching", () => runRelationships(doc.id, true))}
                className="rounded border border-[#3a3128] px-3 py-1.5 text-xs font-semibold tracking-wider hover:border-accent hover:text-accent disabled:opacity-40"
              >
                MATCH
              </button>
            </div>
          </div>
        );
      })}
      {busy && <div className="mt-3 text-xs text-accent">{busy}…</div>}
      {docs.length === 0 && !error && (
        <div className="surface p-6 text-center text-sm text-muted">
          No documents yet. Upload a PDF to start.
        </div>
      )}
    </div>
  );
}
