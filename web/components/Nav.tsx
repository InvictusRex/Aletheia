"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  ["/", "WORKSPACE"],
  ["/facts", "FACTS"],
  ["/relationships", "RELATIONSHIPS"],
] as const;

export function Nav() {
  const path = usePathname();
  return (
    <header className="relative z-20 mx-auto mb-6 flex max-w-[1480px] items-end justify-between border-b border-line px-6 pt-5 pb-3">
      <div>
        <div className="text-[22px] font-bold tracking-[0.22em]">ALETHEIA</div>
        <div className="label mt-1">Fact knowledge layer</div>
      </div>
      <nav className="flex gap-6 text-xs font-semibold tracking-[0.14em]">
        {LINKS.map(([href, label]) => {
          const active = href === "/" ? path === "/" : path.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={active ? "text-accent" : "text-muted hover:text-foreground"}
            >
              {label}
            </Link>
          );
        })}
      </nav>
    </header>
  );
}
