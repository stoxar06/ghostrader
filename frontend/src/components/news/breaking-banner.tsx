"use client";

import { useEffect, useState } from "react";
import type { CompanyNewsItem } from "@/lib/api";
import { timeAgo } from "@/lib/api";
import { Badge } from "@/components/ui/badge";

interface Props {
  items: CompanyNewsItem[];
}

/** Auto-rotating "breaking news" banner for company-specific headlines. */
export function BreakingBanner({ items }: Props) {
  const [i, setI] = useState(0);

  // Only treat genuinely-recent items (< 24h) as "breaking".
  const breaking = items.filter((it) => it.published_ts > Date.now() / 1000 - 86400);
  const list = breaking.length > 0 ? breaking : items.slice(0, 8);

  useEffect(() => {
    if (list.length <= 1) return;
    const t = setInterval(() => setI((n) => (n + 1) % list.length), 5000);
    return () => clearInterval(t);
  }, [list.length]);

  if (list.length === 0) return null;
  const it = list[i % list.length];

  return (
    <div className="overflow-hidden rounded-lg border border-red-500/30 bg-red-500/5">
      <div className="flex items-center gap-3 px-4 py-3">
        <span className="flex shrink-0 items-center gap-1.5">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-500 opacity-75" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-red-500" />
          </span>
          <span className="text-xs font-semibold uppercase tracking-wide text-red-600">Breaking</span>
        </span>
        <a
          href={it.link}
          target="_blank"
          rel="noopener noreferrer"
          className="flex min-w-0 flex-1 items-center gap-2 hover:underline"
          key={it.link}
        >
          <Badge variant="outline" className="shrink-0 text-xs">
            {it.symbol}
          </Badge>
          <span className="truncate text-sm font-medium">{it.title}</span>
        </a>
        <span className="hidden shrink-0 text-xs text-muted-foreground sm:inline">
          {it.source} · {timeAgo(it.published_ts)}
        </span>
      </div>
      {list.length > 1 && (
        <div className="flex gap-1 px-4 pb-2">
          {list.slice(0, 12).map((_, idx) => (
            <button
              key={idx}
              aria-label={`Show item ${idx + 1}`}
              onClick={() => setI(idx)}
              className={`h-1 flex-1 rounded-full transition-colors ${
                idx === i % list.length ? "bg-red-500" : "bg-muted"
              }`}
            />
          ))}
        </div>
      )}
    </div>
  );
}
