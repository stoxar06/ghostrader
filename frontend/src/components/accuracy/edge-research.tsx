"use client";

import { api } from "@/lib/api";
import { useApi } from "@/lib/use-api";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

/** Honest out-of-sample research results (edgesearch archive + evolve hall of fame). */
export function EdgeResearch() {
  const { data, error, loading } = useApi(api.edge);

  if (loading) return <Skeleton className="h-40 w-full rounded-lg" />;
  if (error) return <p className="text-sm text-destructive">{error}</p>;

  const arc = data?.edge_archive;
  const auto = data?.auto_search;
  if (!arc && !auto) {
    return (
      <Card>
        <CardContent className="py-4 text-sm text-muted-foreground">
          No research artifacts yet — run <code>python -m src edgesearch</code> or{" "}
          <code>python -m src evolve</code> to populate this section.
        </CardContent>
      </Card>
    );
  }

  const v = auto?.verdict;
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Out-of-sample edge research (edgesearch + evolve)</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {arc && (
          <p className="text-sm">
            Edge archive: <b>{arc.n_winners}</b> configs cleared 60% OOS accuracy —{" "}
            <b className="text-green-600">{arc.n_real_edge} real edge</b>,{" "}
            <b className="text-amber-600">{arc.n_drift_riders} drift riders</b> (60% that is just
            market drift, not skill).
          </p>
        )}

        {auto && (
          <>
            <p className="text-xs text-muted-foreground">
              Self-improving search: {auto.generations} generations · {auto.configs_tried} configs
              tried · {auto.suspects} lookahead suspects quarantined · last stop: {auto.last_stop}
            </p>

            {v && (
              <div
                className={`rounded-lg border px-4 py-3 text-sm ${
                  v.meaningful
                    ? "border-green-600/40 bg-green-600/5"
                    : "border-amber-500/40 bg-amber-500/5"
                }`}
              >
                <b>Verdict:</b> champion z = {v.z} vs z ≈ {v.z_expected_by_luck} expected from luck
                alone after {auto.configs_tried} tries → {v.note}
              </div>
            )}

            {auto.hall_of_fame.length > 0 && (
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Config</TableHead>
                      <TableHead className="text-right">h</TableHead>
                      <TableHead className="text-right">Fitness</TableHead>
                      <TableHead className="text-right">IS edge</TableHead>
                      <TableHead className="text-right">OOS edge</TableHead>
                      <TableHead className="text-right">OOS acc</TableHead>
                      <TableHead className="text-right">OOS n</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {auto.hall_of_fame.slice(0, 5).map((w) => (
                      <TableRow key={w.config}>
                        <TableCell className="max-w-[320px] truncate font-mono text-xs" title={w.config}>
                          {w.config}
                        </TableCell>
                        <TableCell className="text-right">{w.horizon}</TableCell>
                        <TableCell className="text-right tabular-nums">
                          <Badge variant="secondary" className="text-xs">
                            {w.fitness_pp >= 0 ? "+" : ""}
                            {w.fitness_pp.toFixed(2)}pp
                          </Badge>
                        </TableCell>
                        <TableCell className="text-right tabular-nums">{w.is_cond_edge_pp.toFixed(2)}pp</TableCell>
                        <TableCell className="text-right tabular-nums">{w.oos_cond_edge_pp.toFixed(2)}pp</TableCell>
                        <TableCell className="text-right tabular-nums">{(w.oos_accuracy * 100).toFixed(1)}%</TableCell>
                        <TableCell className="text-right tabular-nums">{w.oos_signals.toLocaleString()}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
          </>
        )}

        <p className="text-xs text-muted-foreground">
          Fitness = min(in-sample, out-of-sample) conditional edge — the edge must exist in both
          halves of history to count. And accuracy is not expectancy: every hall-of-fame config so
          far has <b>lost money net of costs</b> (the costed lens is the final gate).
        </p>
      </CardContent>
    </Card>
  );
}
