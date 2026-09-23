"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { EyeLogo } from "@/components/EyeLogo";
import { listDocuments, listFacts } from "@/lib/api";

export default function Landing() {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [stats, setStats] = useState<{ docs: number; facts: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      const [docs, err] = await listDocuments();
      if (err) return setError(err);
      const counts = await Promise.all(
        (docs ?? []).map(async (d) => (await listFacts(d.id))[0]?.length ?? 0),
      );
      setStats({
        docs: docs?.length ?? 0,
        facts: counts.reduce((a, b) => a + b, 0),
      });
    })();
  }, []);

  const go = () => {
    const q = query.trim();
    router.push(q ? `/facts?q=${encodeURIComponent(q)}` : "/facts");
  };

  return (
    <div className="flex min-h-[calc(100vh-4rem)] flex-col items-center justify-center px-4">
      <div className="w-full max-w-2xl space-y-8 text-center">
        <div className="space-y-4">
          <div className="flex justify-center">
            <EyeLogo className="h-32 w-56 text-accent md:h-36 md:w-64" />
          </div>
          <h1 className="text-5xl font-bold tracking-tight text-accent md:text-6xl">
            ALETHEIA
          </h1>
          <p className="text-4xl font-bold tracking-tight text-accent md:text-5xl">
            Fact Knowledge Layer
          </p>
          <p className="text-lg text-muted">
            Every fact grounded in the page it came from
          </p>
        </div>

        <div className="relative">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") go();
            }}
            placeholder="Search facts, entities, metrics..."
            className="w-full rounded-xl border border-accent/30 bg-card px-5 py-4 text-lg outline-none transition-colors placeholder:text-muted focus:border-accent"
          />
        </div>

        {error && <p className="text-sm text-red">{error}</p>}

        {stats && (
          <p className="text-sm text-muted/60">
            {stats.facts.toLocaleString()} facts across {stats.docs} documents
          </p>
        )}
        {!stats && !error && (
          <div className="flex items-center justify-center gap-2 text-muted">
            <div className="h-4 w-4 animate-spin rounded-full border-2 border-accent border-t-transparent" />
            <span>Reading the knowledge layer...</span>
          </div>
        )}
      </div>
    </div>
  );
}
