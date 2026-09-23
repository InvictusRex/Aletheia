"use client";

import Link from "next/link";
import { use, useEffect, useState } from "react";
import { FactCard } from "@/components/FactCard";
import { VerdictPill } from "@/components/Pill";
import { getRelationship, type RelationshipDetail } from "@/lib/api";

export default function RelationshipPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const [detail, setDetail] = useState<RelationshipDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      const [d, err] = await getRelationship(id);
      setError(err);
      setDetail(d);
    })();
  }, [id]);

  if (error) {
    return <div className="surface border-l-2 border-l-red p-3 text-sm">{error}</div>;
  }
  if (!detail) return <div className="text-sm text-accent">Loading</div>;

  const { relationship: rel, fact_a, fact_b, evidence_a, evidence_b } = detail;
  const meta = rel.reasoning_metadata ?? {};

  return (
    <div className="pb-24">
      <Link
        href="/relationships"
        className="mb-3 inline-block text-xs text-muted hover:text-accent"
      >
        &larr; all relationships
      </Link>

      <div className="surface mb-4 p-4">
        <VerdictPill rel={rel} />
        <p className="mt-3 text-sm leading-relaxed text-[#b5b5b5]">
          {rel.explanation}
        </p>
      </div>

      <div className="mb-4 grid gap-4 md:grid-cols-2">
        <div>
          <div className="label mb-2">Fact A</div>
          <FactCard fact={fact_a} docName="source A" evidence={evidence_a} />
        </div>
        <div>
          <div className="label mb-2">Fact B</div>
          <FactCard fact={fact_b} docName="source B" evidence={evidence_b} />
        </div>
      </div>

      <div className="surface p-4">
        <div className="label mb-2">Reasoning metadata</div>
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs">
          {Object.entries(meta).map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="text-muted">{k}</dt>
              <dd className="break-words">
                {Array.isArray(v) ? v.join(", ") || "none" : String(v)}
              </dd>
            </div>
          ))}
        </dl>
      </div>
    </div>
  );
}
