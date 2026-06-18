"use client";

import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import { useApi } from "@/lib/use-api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { EdgeResearch } from "@/components/accuracy/edge-research";

type Tab = "single" | "volume" | "confluence";

interface UiRow {
  name: string;
  info: string;
  accuracy: number; // 0..1
  edge_pp: number;
  signals: number;
  detail: string; // coverage / delta line
}

const TABS: { key: Tab; label: string }[] = [
  { key: "single", label: "Single indicators" },
  { key: "volume", label: "+ Volume" },
  { key: "confluence", label: "Confluence + regime" },
];

type Sig = "above" | "below" | "noise";

/** Is an edge real or just sampling noise? Compares it to ~2 standard errors of a
 * proportion (p≈0.5 upper bound). A +0.1pp "edge" on thousands of signals is noise. */
function significance(edgePp: number, signals: number): Sig {
  if (!signals) return "noise";
  const twoSE = 2 * Math.sqrt(0.25 / signals) * 100; // in percentage points
  if (edgePp > twoSE) return "above";
  if (edgePp < -twoSE) return "below";
  return "noise";
}

export default function AccuracyPage() {
  const { data, error, loading, refetch } = useApi((s) => api.indicatorAccuracy(5, s));
  const [tab, setTab] = useState<Tab>("single");
  const [selected, setSelected] = useState<string | null>(null);

  // Persist the user's chosen indicator.
  useEffect(() => {
    setSelected(localStorage.getItem("ghost.indicator"));
  }, []);
  const choose = (name: string) => {
    setSelected(name);
    localStorage.setItem("ghost.indicator", name);
  };

  const base = data?.base ?? 0;

  const rows: UiRow[] = useMemo(() => {
    if (!data) return [];
    if (tab === "single")
      return data.single.map((r) => ({
        name: r.indicator,
        info: r.info,
        accuracy: r.accuracy,
        edge_pp: r.edge_pp,
        signals: r.signals,
        detail: `coverage ${(r.coverage * 100).toFixed(0)}% · ${r.signals.toLocaleString()} signals`,
      }));
    if (tab === "confluence")
      return data.confluence.map((r) => ({
        name: r.rule,
        info: r.info,
        accuracy: r.accuracy,
        edge_pp: r.edge_pp,
        signals: r.signals,
        detail: `coverage ${(r.coverage * 100).toFixed(0)}% · ${r.signals.toLocaleString()} signals`,
      }));
    return data.volume.map((r) => ({
      name: r.indicator,
      info: r.info,
      accuracy: r.vol_acc,
      edge_pp: r.vol_edge_pp,
      signals: r.vol_signals,
      detail: `${r.delta_pp >= 0 ? "+" : ""}${r.delta_pp.toFixed(1)} pp from volume · ${r.vol_signals.toLocaleString()} signals`,
    }));
  }, [data, tab]);

  const anyEdge = rows.some((r) => significance(r.edge_pp, r.signals) === "above");
  const selectedRow = rows.find((r) => r.name === selected);
  const selectedSig = selectedRow ? significance(selectedRow.edge_pp, selectedRow.signals) : "noise";

  return (
    <main className="mx-auto w-full max-w-7xl flex-1 space-y-6 px-4 py-6 sm:px-6">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Indicator accuracy</h1>
          <p className="text-sm text-muted-foreground">
            Directional hit-rate measured on your data · pick an indicator to use
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={refetch} disabled={loading}>
          {loading ? "Measuring…" : "Re-measure"}
        </Button>
      </div>

      {/* Base-rate context */}
      {loading ? (
        <Skeleton className="h-20 w-full rounded-lg" />
      ) : error ? (
        <p className="text-sm text-destructive">{error}</p>
      ) : data ? (
        <Card>
          <CardContent className="flex flex-wrap items-center gap-x-8 gap-y-3 py-4">
            <Stat label="Base rate (to beat)" value={`${(base * 100).toFixed(1)}%`} hint="market drift" />
            <Stat label="Horizon" value={`${data.horizon} bars`} />
            <Stat label="Universe" value={`${data.symbols} stocks`} hint={data.timeframe} />
            <p className="ml-auto max-w-md text-xs text-muted-foreground">
              An indicator only has real edge if its accuracy is <b>above the base rate</b> —
              the score you&apos;d get by always betting on the market&apos;s upward drift.
            </p>
          </CardContent>
        </Card>
      ) : null}

      {/* Tabs */}
      <div className="flex flex-wrap gap-1">
        {TABS.map((t) => (
          <Button
            key={t.key}
            variant={tab === t.key ? "default" : "outline"}
            size="sm"
            onClick={() => setTab(t.key)}
          >
            {t.label}
          </Button>
        ))}
      </div>

      {!loading && !error && data && !anyEdge && (
        <div className="rounded-lg border border-amber-500/40 bg-amber-500/5 px-4 py-3 text-sm">
          ⚠️ In this view, <b>no option beats the {(base * 100).toFixed(1)}% base rate by a
          statistically significant margin</b> — these indicators don&apos;t add durable directional
          edge. A reading like 53.6% vs 53.5% is <b>noise</b> (within sampling error), not an edge.
          You can still select one, but expect ~coin-flip direction before costs.
        </div>
      )}

      {/* Selectable list */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="space-y-2 lg:col-span-2">
          {loading
            ? Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-16 w-full rounded-lg" />)
            : rows.map((r) => {
                const sig = significance(r.edge_pp, r.signals);
                const isSel = r.name === selected;
                return (
                  <button
                    key={r.name}
                    onClick={() => choose(r.name)}
                    className={`flex w-full items-center gap-4 rounded-lg border px-4 py-3 text-left transition-colors hover:bg-accent ${
                      isSel ? "border-primary ring-1 ring-primary" : ""
                    }`}
                  >
                    <span
                      className={`flex h-4 w-4 shrink-0 items-center justify-center rounded-full border ${
                        isSel ? "border-primary bg-primary" : "border-muted-foreground/40"
                      }`}
                    >
                      {isSel && <span className="h-1.5 w-1.5 rounded-full bg-primary-foreground" />}
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="flex items-center gap-2">
                        <span className="font-medium">{r.name}</span>
                        {sig === "above" ? (
                          <Badge className="bg-green-600 text-xs hover:bg-green-600">beats base</Badge>
                        ) : sig === "below" ? (
                          <Badge variant="secondary" className="text-xs">
                            below base
                          </Badge>
                        ) : (
                          <Badge
                            variant="outline"
                            className="border-amber-500/50 text-xs text-amber-600"
                            title="Within sampling error — not a real edge"
                          >
                            ≈ base (noise)
                          </Badge>
                        )}
                      </span>
                      <span className="line-clamp-1 text-xs text-muted-foreground">{r.info}</span>
                      <span className="text-xs text-muted-foreground">{r.detail}</span>
                    </span>
                    <span className="shrink-0 text-right">
                      <span className="block text-lg font-semibold tabular-nums">
                        {(r.accuracy * 100).toFixed(1)}%
                      </span>
                      <span
                        className={`block text-xs tabular-nums ${
                          r.edge_pp >= 0 ? "text-green-600" : "text-red-600"
                        }`}
                      >
                        {r.edge_pp >= 0 ? "+" : ""}
                        {r.edge_pp.toFixed(1)} pp
                      </span>
                    </span>
                  </button>
                );
              })}
        </div>

        {/* Selection summary */}
        <Card className="h-fit">
          <CardHeader>
            <CardTitle className="text-sm font-medium text-muted-foreground">Your selection</CardTitle>
          </CardHeader>
          <CardContent>
            {!selected ? (
              <p className="text-sm text-muted-foreground">Pick an indicator from the list.</p>
            ) : selectedRow ? (
              <div className="space-y-3">
                <div>
                  <p className="text-lg font-semibold">{selectedRow.name}</p>
                  <p className="text-sm text-muted-foreground">{selectedRow.info}</p>
                </div>
                <div className="flex gap-6">
                  <Stat label="Accuracy" value={`${(selectedRow.accuracy * 100).toFixed(1)}%`} />
                  <Stat
                    label="Edge vs base"
                    value={`${selectedRow.edge_pp >= 0 ? "+" : ""}${selectedRow.edge_pp.toFixed(1)} pp`}
                  />
                </div>
                <p className="text-xs text-muted-foreground">{selectedRow.detail}</p>
                {selectedSig !== "above" && (
                  <p className="rounded-md bg-amber-500/10 px-3 py-2 text-xs">
                    Heads-up: {selectedSig === "noise"
                      ? `+${selectedRow.edge_pp.toFixed(1)}pp over the ${(base * 100).toFixed(1)}% base rate is within sampling error — statistically it's a coin flip, not an edge.`
                      : `this sits below the ${(base * 100).toFixed(1)}% base rate.`}{" "}
                    And measured as a real strategy (net of costs, with stops) it <b>loses money</b> —
                    accuracy alone doesn&apos;t pay; expectancy does.
                  </p>
                )}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                &quot;{selected}&quot; isn&apos;t in this tab — switch tabs to see it.
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Honest OOS research artifacts (edgesearch + the self-improving evolve loop) */}
      <EdgeResearch />
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
