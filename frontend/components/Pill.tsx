import type { Fact, Relationship } from "@/lib/api";

export function Pill({ tone = "muted", children }: { tone?: "accent" | "green" | "red" | "muted"; children: React.ReactNode }) {
  return <span className={`pill pill-${tone}`}>{children}</span>;
}

/** What a fact can take part in, and what it is missing if not. */
export function ComparabilityPill({ fact }: { fact: Fact }) {
  if (fact.comparability === "COMPARABLE") return <Pill tone="green">COMPARABLE</Pill>;
  const missing = fact.missing_for_comparison.join(", ");
  return fact.comparability === "PARTIAL" ? (
    <Pill tone="accent">PARTIAL: no {missing}</Pill>
  ) : (
    <Pill tone="red">NOT COMPARABLE: no {missing}</Pill>
  );
}

const TONE: Record<string, "green" | "red" | "muted"> = {
  CORROBORATES: "green",
  CONTRADICTS: "red",
};

export function VerdictPill({ rel }: { rel: Relationship }) {
  const withheld = rel.reasoning_metadata?.["withheld_verdict"] as string | undefined;
  const reason = rel.reasoning_metadata?.["withheld_reason"] as string | undefined;
  return (
    <span className="inline-flex flex-wrap items-center gap-2">
      <Pill tone={TONE[rel.relationship_type] ?? "muted"}>{rel.relationship_type}</Pill>
      {rel.status === "NEEDS_REVIEW" && <Pill tone="accent">NEEDS REVIEW</Pill>}
      <span className="text-xs text-muted">{(rel.confidence * 100).toFixed(0)}%</span>
      {withheld && (
        <span className="text-xs text-muted">
          · {withheld} withheld: {reason ?? "precondition not met"}
        </span>
      )}
    </span>
  );
}
