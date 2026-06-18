"use client";

import { api } from "@/lib/api";
import { useApi } from "@/lib/use-api";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Panel } from "./panel";
import { PnlHeatmap } from "@/components/pnl-heatmap";

const inr = (n: number) =>
  `${n >= 0 ? "+" : "−"}₹${Math.abs(n).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;

/** Mini per-day P&L bars, oldest → newest (green up, red down). */
function DailyBars({ rows }: { rows: { day: string; realized_pnl: number }[] }) {
  const chrono = [...rows].reverse();
  const max = Math.max(...chrono.map((r) => Math.abs(r.realized_pnl)), 1);
  return (
    <div className="mb-4 flex h-24 items-end gap-[2px]" aria-label="Daily P&L bars">
      {chrono.map((r) => (
        <div
          key={r.day}
          title={`${r.day}: ${inr(r.realized_pnl)}`}
          className={`min-w-[6px] flex-1 rounded-sm ${
            r.realized_pnl >= 0 ? "bg-green-600/80" : "bg-red-600/80"
          }`}
          style={{ height: `${Math.max(6, (Math.abs(r.realized_pnl) / max) * 100)}%` }}
        />
      ))}
    </div>
  );
}

export function PnlCard() {
  const { data, error, loading, refetch } = useApi(api.dailyPnl);
  return (
    <Panel title="Daily realized P&L (last 30 recorded days)" loading={loading} error={error} onRefresh={refetch} className="lg:col-span-2">
      {data && data.length === 0 && <p className="text-sm text-muted-foreground">No P&L recorded yet.</p>}
      {data && data.length > 0 && (
        <div className="mb-4">
          <PnlHeatmap
            days={data.map((r) => ({
              date: r.day,
              pnl: r.realized_pnl,
              label: `${r.trades} trade${r.trades === 1 ? "" : "s"}`,
            }))}
            format={inr}
          />
        </div>
      )}
      {data && data.length > 1 && <DailyBars rows={data} />}
      {data && data.length > 0 && (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Day</TableHead>
              <TableHead className="text-right">P&L</TableHead>
              <TableHead className="text-right">Trades</TableHead>
              <TableHead>Mode</TableHead>
              <TableHead className="text-right">Halted</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {data.map((r) => (
              <TableRow key={r.day}>
                <TableCell className="font-medium">{r.day}</TableCell>
                <TableCell className={`text-right ${r.realized_pnl >= 0 ? "text-green-600" : "text-red-600"}`}>
                  {inr(r.realized_pnl)}
                </TableCell>
                <TableCell className="text-right">{r.trades}</TableCell>
                <TableCell>
                  <Badge variant="secondary" className="text-xs">
                    {r.mode}
                  </Badge>
                </TableCell>
                <TableCell className="text-right">
                  {r.halted ? <Badge variant="destructive">halted</Badge> : "—"}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </Panel>
  );
}
