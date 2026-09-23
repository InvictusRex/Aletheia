"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { FactCard } from "@/components/FactCard";
import {
  getBundle,
  listDocuments,
  listFacts,
  search,
  type Doc,
  type Evidence,
  type Fact,
} from "@/lib/api";

function FactsInner() {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [facts, setFacts] = useState<Fact[]>([]);
  const [evidence, setEvidence] = useState<Evidence[]>([]);
  const params = useSearchParams();
  const [query, setQuery] = useState(params.get("q") ?? "");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      const [list, err] = await listDocuments();
      if (err) return setError(err);
      setDocs(list ?? []);
      if (list?.length) setSelected(list[0].id);
    })();
  }, []);

  useEffect(() => {
    const incoming = params.get("q");
    if (incoming && selected) void runSearch(incoming);
    // Only when the landing page hands us a query.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected]);

  useEffect(() => {
    if (!selected) return;
    if (params.get("q")) return;
    void (async () => {
      setLoading(true);
      const [f, fe] = await listFacts(selected);
      const [b] = await getBundle(selected);
      setError(fe);
      setFacts(f ?? []);
      setEvidence(b?.evidence ?? []);
      setLoading(false);
    })();
  }, [selected]);

  const runSearch = async (override?: string) => {
    const q = (override ?? query).trim();
    if (!q) return;
    setLoading(true);
    const [res, err] = await search(q, selected || undefined);
    setError(err);
    setFacts((res?.hits ?? []).map((h) => h.fact));
    setEvidence((res?.hits ?? []).flatMap((h) => h.evidence));
    setLoading(false);
  };

  const docName = docs.find((d) => d.id === selected)?.filename ?? "source";
  const counts = facts.reduce<Record<string, number>>((acc, f) => {
    acc[f.comparability] = (acc[f.comparability] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <div className="pb-24">
      <div className="surface mb-4 flex flex-wrap items-center gap-3 p-3">
        <select
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          className="surface min-w-[260px] px-2 py-1.5 text-sm"
        >
          {docs.map((d) => (
            <option key={d.id} value={d.id}>
              {d.filename}
            </option>
          ))}
        </select>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void runSearch();
          }}
          placeholder="Search facts, entities, metrics"
          className="surface min-w-[260px] flex-1 px-3 py-1.5 text-sm"
        />
        <button
          onClick={() => void runSearch()}
          className="rounded border border-[#3a3128] px-3 py-1.5 text-xs font-semibold tracking-wider hover:border-accent hover:text-accent"
        >
          SEARCH
        </button>
      </div>

      <div className="mb-3 text-xs text-muted">
        {facts.length} fact(s)
        {counts.COMPARABLE ? ` · ${counts.COMPARABLE} comparable` : ""}
        {counts.PARTIAL ? ` · ${counts.PARTIAL} partial` : ""}
        {counts.NOT_COMPARABLE ? ` · ${counts.NOT_COMPARABLE} not comparable` : ""}
      </div>

      {error && (
        <div className="surface mb-4 border-l-2 border-l-red p-3 text-sm">{error}</div>
      )}
      {loading && <div className="text-sm text-accent">Loading</div>}

      {facts.slice(0, 200).map((f) => (
        <FactCard key={f.id} fact={f} docName={docName} evidence={evidence} />
      ))}
    </div>
  );
}


export default function Facts() {
  return (
    <Suspense fallback={<div className="text-sm text-accent">Loading</div>}>
      <FactsInner />
    </Suspense>
  );
}
