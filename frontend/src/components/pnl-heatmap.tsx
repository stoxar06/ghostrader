"use client";

import { useMemo } from "react";

export interface HeatDay {
  date: string; // YYYY-MM-DD
  pnl: number;
  label?: string; // tooltip detail, e.g. "+0.42% · 3 trades"
}

const DAY_MS = 86_400_000;
const WEEKDAYS = ["", "Mon", "", "Wed", "", "Fri", ""];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function utc(date: string): number {
  const [y, m, d] = date.split("-").map(Number);
  return Date.UTC(y, m - 1, d);
}

function cellColor(pnl: number, max: number): string {
  if (pnl === 0) return "";
  const level = Math.min(1, Math.abs(pnl) / max);
  const alpha = 0.25 + 0.75 * level;
  return pnl > 0 ? `rgba(34, 197, 94, ${alpha})` : `rgba(239, 68, 68, ${alpha})`;
}

/** GitHub-contributions-style calendar of daily P&L: columns are weeks, rows are
 * weekdays, green = profit day, red = loss day, intensity = size of the move. */
export function PnlHeatmap({ days, format = (n) => n.toFixed(2) }: {
  days: HeatDay[];
  format?: (pnl: number) => string;
}) {
  const { weeks, monthLabels, max } = useMemo(() => {
    const byDate = new Map(days.map((d) => [utc(d.date), d]));
    const stamps = [...byDate.keys()];
    const first = Math.min(...stamps);
    const last = Math.max(...stamps);
    const start = first - ((new Date(first).getUTCDay()) * DAY_MS); // back to Sunday
    const max = Math.max(...days.map((d) => Math.abs(d.pnl)), 1e-9);

    const weeks: (HeatDay | null)[][] = [];
    const monthLabels: string[] = [];
    for (let w = start; w <= last; w += 7 * DAY_MS) {
      const col: (HeatDay | null)[] = [];
      for (let i = 0; i < 7; i++) col.push(byDate.get(w + i * DAY_MS) ?? null);
      weeks.push(col);
      const d = new Date(w);
      monthLabels.push(d.getUTCDate() <= 7 ? MONTHS[d.getUTCMonth()] : "");
    }
    return { weeks, monthLabels, max };
  }, [days]);

  if (!days.length) return null;

  return (
    <div className="overflow-x-auto">
      <div className="inline-flex flex-col gap-1">
        <div className="ml-8 flex gap-[3px] text-[10px] text-muted-foreground">
          {monthLabels.map((m, i) => (
            <span key={i} className="w-[11px] shrink-0">{m}</span>
          ))}
        </div>
        <div className="flex gap-1">
          <div className="flex w-7 flex-col gap-[3px] text-[10px] leading-[11px] text-muted-foreground">
            {WEEKDAYS.map((d, i) => (
              <span key={i} className="h-[11px]">{d}</span>
            ))}
          </div>
          <div className="flex gap-[3px]">
            {weeks.map((col, wi) => (
              <div key={wi} className="flex flex-col gap-[3px]">
                {col.map((d, di) => (
                  <div
                    key={di}
                    title={d ? `${d.date}: ${format(d.pnl)}${d.label ? ` · ${d.label}` : ""}` : undefined}
                    className={`h-[11px] w-[11px] rounded-[2px] ${d ? "" : "bg-muted/40"}`}
                    style={d ? { backgroundColor: cellColor(d.pnl, max) || "var(--muted)" } : undefined}
                  />
                ))}
              </div>
            ))}
          </div>
        </div>
        <div className="ml-8 flex items-center gap-1 text-[10px] text-muted-foreground">
          <span>loss</span>
          {[-1, -0.5, 0.5, 1].map((v) => (
            <span
              key={v}
              className="h-[11px] w-[11px] rounded-[2px]"
              style={{ backgroundColor: cellColor(v, 1) }}
            />
          ))}
          <span>profit</span>
        </div>
      </div>
    </div>
  );
}
