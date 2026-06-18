"use client";

import { useMemo, useState } from "react";
import { api } from "@/lib/api";
import { useApi } from "@/lib/use-api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Panel } from "./panel";

const num = (n: number | null) => (n == null ? "—" : n.toLocaleString(undefined, { maximumFractionDigits: 2 }));
const inr = (n: number) =>
  `${n >= 0 ? "+" : "−"}₹${Math.abs(n).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;

export function TradesCard() {
  // "Last 7 days" = the 7 most recent *recorded* trading days (replay sessions are
  // historical, so a calendar-today window would usually be empty).
  const [days, setDays] = useState<number | undefined>(7);
  const { data, error, loading, refetch } = useApi((s) => api.trades(days, s), [days]);

  const byDay = useMemo(() => {
    const m = new Map<string, { pnl: number; n: number }>();
    for (const t of data ?? []) {
      const d = t.day ?? "—";
      const cur = m.get(d) ?? { pnl: 0, n: 0 };
      cur.pnl += t.pnl ?? 0;
      cur.n += 1;
      m.set(d, cur);
    }
    return [...m.entries()].sort((a, b) => (a[0] < b[0] ? 1 : -1));
  }, [data]);

  return (
    <Panel
      title={days ? `Trades — last ${days} recorded days` : "Recent trades"}
      loading={loading}
      error={error}
      onRefresh={refetch}
      className="lg:col-span-3"
    >
      <div className="mb-3 flex flex-wrap gap-1">
        <Button variant={days === 7 ? "default" : "outline"} size="sm" onClick={() => setDays(7)}>
          Last 7 days
        </Button>
        <Button variant={days == null ? "default" : "outline"} size="sm" onClick={() => setDays(undefined)}>
          All recent
        </Button>
      </div>

      {days != null && byDay.length > 0 && (
        <div className="mb-3 flex flex-wrap gap-2">
          {byDay.map(([d, v]) => (
            <div key={d} className="rounded-md border px-3 py-1.5 text-xs">
              <span className="text-muted-foreground">{d}</span>{" "}
              <span className={v.pnl >= 0 ? "text-green-600" : "text-red-600"}>{inr(v.pnl)}</span>{" "}
              <span className="text-muted-foreground">· {v.n} trade{v.n === 1 ? "" : "s"}</span>
            </div>
          ))}
        </div>
      )}

      {data && data.length === 0 && (
        <p className="text-sm text-muted-foreground">
          {days ? `No trades in the last ${days} recorded days.` : "No trades yet."}
        </p>
      )}
      {data && data.length > 0 && (
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Day</TableHead>
                <TableHead>Symbol</TableHead>
                <TableHead>Side</TableHead>
                <TableHead className="text-right">Qty</TableHead>
                <TableHead className="text-right">Entry</TableHead>
                <TableHead className="text-right">Exit</TableHead>
                <TableHead className="text-right">P&L</TableHead>
                <TableHead>Reason</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.map((t, i) => (
                <TableRow key={i}>
                  <TableCell className="whitespace-nowrap text-muted-foreground">{t.day ?? "—"}</TableCell>
                  <TableCell className="font-medium">{t.symbol}</TableCell>
                  <TableCell>
                    <Badge variant={t.side?.toLowerCase() === "buy" ? "default" : "secondary"} className="text-xs">
                      {t.side}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-right">{t.qty}</TableCell>
                  <TableCell className="text-right">{num(t.entry)}</TableCell>
                  <TableCell className="text-right">{num(t.exit)}</TableCell>
                  <TableCell
                    className={`text-right ${
                      t.pnl == null ? "" : t.pnl >= 0 ? "text-green-600" : "text-red-600"
                    }`}
                  >
                    {num(t.pnl)}
                  </TableCell>
                  <TableCell className="max-w-[200px] truncate text-muted-foreground">{t.reason}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </Panel>
  );
}
