"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { VerdictPill } from "@/components/Pill";
import { listDocuments, listRelationships, type Relationship } from "@/lib/api";

const TYPES = [
  "ALL",
  "CONTRADICTS",
  "CORROBORATES",
  "CONTEXTUAL_DIFFERENCE",
  "RELATED",
];

/** A contradiction outranks two thousand contextual notes. */
const RANK: Record<string, number> = {
  CONTRADICTS: 0,
  CORROBORATES: 1,
  RELATED: 2,
};

export default function Relationships() {
  const [rels, setRels] = useState<Relationship[]>([]);
  const [filter, setFilter] = useState("ALL");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    void (async () => {
      const [docs] = await listDocuments();
      const all: Relationship[] = [];
      const seen = new Set<string>();
      for (const doc of docs ?? []) {
        const [list] = await listRelationships(doc.id);
        for (const r of list ?? []) {
          if (seen.has(r.id)) continue;
          seen.add(r.id);
          all.push(r);
        }
      }
      all.sort(
        (a, b) =>
          (RANK[a.relationship_type] ?? 3) - (RANK[b.relationship_type] ?? 3) ||
          b.confidence - a.confidence,
      );
      setRels(all);
      setLoading(false);
    })();
  }, []);

  const shown =
    filter === "ALL" ? rels : rels.filter((r) => r.relationship_type === filter);
  const counts = rels.reduce<Record<string, number>>((acc, r) => {
    acc[r.relationship_type] = (acc[r.relationship_type] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <div>
      <div className="mb-4 flex flex-wrap gap-2">
        {TYPES.map((t) => (
          <button
            key={t}
            onClick={() => setFilter(t)}
            className={`rounded border px-3 py-1.5 text-xs font-semibold tracking-wider ${
              filter === t
                ? "border-accent text-accent"
                : "border-[#3a3128] text-muted hover:text-foreground"
            }`}
          >
            {t}
            {t !== "ALL" && counts[t] ? ` (${counts[t]})` : ""}
          </button>
        ))}
      </div>

      {loading && <div className="text-sm text-accent">Loading</div>}
      {!loading && shown.length === 0 && (
        <div className="surface p-6 text-center text-sm text-muted">
          No relationships of this type. Normalize and match first.
        </div>
      )}

      {shown.slice(0, 300).map((r) => (
        <Link
          key={r.id}
          href={`/relationships/${r.id}`}
          className="surface mb-2 block p-4 hover:border-accent"
        >
          <VerdictPill rel={r} />
          <div className="mt-2 line-clamp-3 text-sm text-[#b5b5b5]">
            {r.explanation}
          </div>
        </Link>
      ))}
    </div>
  );
}
