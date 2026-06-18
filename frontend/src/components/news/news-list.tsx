"use client";

import type { CompanyNewsItem, NewsItem } from "@/lib/api";
import { timeAgo } from "@/lib/api";
import { Badge } from "@/components/ui/badge";

function isCompany(it: NewsItem | CompanyNewsItem): it is CompanyNewsItem {
  return "symbol" in it;
}

export function NewsList({ items }: { items: (NewsItem | CompanyNewsItem)[] }) {
  if (items.length === 0) return <p className="text-sm text-muted-foreground">No news.</p>;
  return (
    <ul className="divide-y">
      {items.map((it, i) => (
        <li key={`${it.link}-${i}`} className="py-3">
          <a
            href={it.link}
            target="_blank"
            rel="noopener noreferrer"
            className="group flex flex-col gap-1"
          >
            <span className="flex items-start gap-2">
              {isCompany(it) && (
                <Badge variant="secondary" className="mt-0.5 shrink-0 text-xs">
                  {it.symbol}
                </Badge>
              )}
              <span className="text-sm font-medium leading-snug group-hover:underline">
                {it.title}
              </span>
            </span>
            <span className="text-xs text-muted-foreground">
              {it.source}
              {it.published_ts ? ` · ${timeAgo(it.published_ts)}` : ""}
            </span>
          </a>
        </li>
      ))}
    </ul>
  );
}
