"use client";

import { useMemo, useState } from "react";
import { api } from "@/lib/api";
import { useApi } from "@/lib/use-api";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { PnlHeatmap } from "@/components/pnl-heatmap";

const pct = (n: number, digits = 1) => `${n >= 0 ? "+" : ""}${n.toFixed(digits)}%`;

export default function HorizonPage() {
  const { data, error, loading, refetch } = useApi((s) => api.horizon(12, s));
  const [selected, setSelected] = useState<string | null>(null);

  const horizons = useMemo(
    () => (data ? Object.keys(data.horizons).sort((a, b) => Number(a) - Number(b)) : []),
    [data],
  );
  // Default to the horizon with the best total P&L.
  const active = selected ?? (data
    ? horizons.reduce((best, h) =>
        data.horizons[h].total_pnl_pct > data.horizons[best].total_pnl_pct ? h : best, horizons[0])
    : null);
  const daily = data && active ? data.daily[active] : null;

  return (
    <main className="mx-auto w-full max-w-7xl flex-1 space-y-6 px-4 py-6 sm:px-6">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Holding-horizon P&L</h1>
          <p className="text-sm text-muted-foreground">
            Every signal held exactly 1–12 days · daily profit/loss + totals, net of costs
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={refetch} disabled={loading}>
          {loading ? "Measuring…" : "Re-measure"}
        </Button>
      </div>

      {loading ? (
        <>
          <Skeleton className="h-20 w-full rounded-lg" />
          <Skeleton className="h-72 w-full rounded-lg" />
        </>
      ) : error ? (
        <p className="text-sm text-destructive">{error}</p>
      ) : data ? (
        <>
          <Card>
            <CardContent className="flex flex-wrap items-center gap-x-8 gap-y-3 py-4">
              <Stat label="Universe" value={`${data.symbols} stocks`} hint={data.timeframe} />
              <Stat label="Round-trip cost" value={`${(data.cost_pct * 100).toFixed(2)}%`} />
              <Stat label="Horizons" value={`1–${horizons[horizons.length - 1]} days`} />
              <p className="ml-auto max-w-md text-xs text-muted-foreground">
                A horizon only has real edge if its win-rate beats the <b>base up-rate</b> AND
                total P&L stays positive after costs. Long holds that win by riding market
                drift are the equity risk premium, not a signal.
              </p>
            </CardContent>
          </Card>

          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Hold (days)</TableHead>
                <TableHead className="text-right">Signals</TableHead>
                <TableHead className="text-right">Win%</TableHead>
                <TableHead className="text-right">Profit days</TableHead>
                <TableHead className="text-right">Loss days</TableHead>
                <TableHead className="text-right">Avg trade</TableHead>
                <TableHead className="text-right">Total P&L</TableHead>
                <TableHead className="text-right">Base up-rate</TableHead>
                <TableHead className="text-right">Edge vs base</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {horizons.map((h) => {
                const r = data.horizons[h];
                const isSel = h === active;
                return (
                  <TableRow
                    key={h}
                    onClick={() => setSelected(h)}
                    className={`cursor-pointer ${isSel ? "bg-accent" : ""}`}
                  >
                    <TableCell className="font-medium">
                      {h}d {isSel && <Badge variant="secondary" className="ml-1 text-xs">shown below</Badge>}
                    </TableCell>
                    <TableCell className="text-right">{r.signals.toLocaleString()}</TableCell>
                    <TableCell className="text-right">{(r.win_rate * 100).toFixed(1)}</TableCell>
                    <TableCell className="text-right text-green-600">{r.profit_days ?? "—"}</TableCell>
                    <TableCell className="text-right text-red-600">{r.loss_days ?? "—"}</TableCell>
                    <TableCell className={`text-right ${r.avg_ret_pct >= 0 ? "text-green-600" : "text-red-600"}`}>
                      {pct(r.avg_ret_pct, 2)}
                    </TableCell>
                    <TableCell className={`text-right font-medium ${r.total_pnl_pct >= 0 ? "text-green-600" : "text-red-600"}`}>
                      {pct(r.total_pnl_pct)}
                    </TableCell>
                    <TableCell className="text-right">{(r.base_up_rate * 100).toFixed(1)}%</TableCell>
                    <TableCell className={`text-right ${r.edge_vs_base > 0 ? "text-green-600" : "text-red-600"}`}>
                      {pct(r.edge_vs_base * 100)} pp
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>

          {daily && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm font-medium">
                  Daily P&L at a {active}-day hold — GitHub-style calendar
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <PnlHeatmap
                  days={daily.days.map((d) => ({
                    date: d.date,
                    pnl: d.pnl_pct,
                    label: `${d.trades} trade${d.trades === 1 ? "" : "s"} closed`,
                  }))}
                  format={(n) => pct(n, 2)}
                />
                <div className="flex flex-wrap gap-6 border-t pt-4">
                  <Stat label="Profit days" value={String(daily.profit_days)} />
                  <Stat label="Loss days" value={String(daily.loss_days)} />
                  <Stat label="Total P&L (sum of daily)" value={pct(daily.total_pnl_pct)} />
                </div>
              </CardContent>
            </Card>
          )}
        </>
      ) : null}
    </main>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div>
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-lg font-semibold tabular-nums">
        {value} {hint && <span className="text-xs font-normal text-muted-foreground">{hint}</span>}
      </p>
    </div>
  );
}
