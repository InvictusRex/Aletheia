"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { EyeLogo } from "@/components/EyeLogo";

const LINKS = [
  { href: "/workspace", label: "Workspace" },
  { href: "/facts", label: "Facts" },
  { href: "/relationships", label: "Relationships" },
];

export function Nav() {
  const path = usePathname();
  return (
    <nav className="fixed inset-x-0 top-0 z-50 border-b border-accent/30 bg-black/80 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-[90rem] items-center justify-between px-4">
        <Link
          href="/"
          className="flex items-center gap-2 text-accent transition-colors hover:text-accent/80"
        >
          <EyeLogo className="h-9 w-16" />
          <span className="text-xl font-bold tracking-[0.18em]">ALETHEIA</span>
        </Link>
        <div className="flex items-center gap-1">
          {LINKS.map(({ href, label }) => {
            const active = path === href || path.startsWith(`${href}/`);
            return (
              <Link
                key={href}
                href={href}
                className={`rounded-lg px-4 py-2 text-[15px] font-semibold transition-all duration-200 ${
                  active
                    ? "bg-accent text-black"
                    : "text-foreground/70 hover:bg-accent/10 hover:text-accent"
                }`}
              >
                {label}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
