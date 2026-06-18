"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ThemeToggle } from "@/components/theme-toggle";

const LINKS = [
  { href: "/", label: "Dashboard" },
  { href: "/accuracy", label: "Accuracy" },
  { href: "/horizon", label: "Horizon P&L" },
  { href: "/news", label: "News" },
];

export function Nav() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-20 border-b bg-background/80 backdrop-blur">
      <nav className="mx-auto flex h-12 w-full max-w-7xl items-center gap-6 px-4 sm:px-6">
        <span className="font-semibold tracking-tight">Ghostrader</span>
        <div className="flex items-center gap-4 text-sm">
          {LINKS.map((l) => {
            const active = l.href === "/" ? pathname === "/" : pathname.startsWith(l.href);
            return (
              <Link
                key={l.href}
                href={l.href}
                className={
                  active
                    ? "font-medium text-foreground"
                    : "text-muted-foreground transition-colors hover:text-foreground"
                }
              >
                {l.label}
              </Link>
            );
          })}
        </div>
        <div className="ml-auto">
          <ThemeToggle />
        </div>
      </nav>
    </header>
  );
}
