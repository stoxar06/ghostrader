"use client";

import { api } from "@/lib/api";
import { useApi } from "@/lib/use-api";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Panel } from "./panel";

function biasVariant(bias: string): "default" | "destructive" | "secondary" {
  if (bias === "bullish" || bias === "risk_on") return "default";
  if (bias === "bearish" || bias === "risk_off") return "destructive";
  return "secondary";
}

export function BriefingCard() {
  const { data, error, loading, refetch } = useApi(api.briefing);
  return (
    <Panel title="Market briefing" loading={loading} error={error} onRefresh={refetch} className="lg:col-span-2">
      {data && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant={biasVariant(data.regime.regime)}>{data.regime.regime}</Badge>
            <Badge variant={biasVariant(data.regime.bias)}>bias: {data.regime.bias}</Badge>
            <Badge variant="outline" className="text-xs">
              {data.regime.source}
            </Badge>
          </div>
          <p className="text-sm text-muted-foreground">{data.regime.summary}</p>

          {Object.keys(data.cues).length > 0 && (
            <>
              <Separator />
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                {Object.entries(data.cues).map(([name, pct]) => (
                  <div key={name} className="flex items-baseline justify-between gap-2 text-sm">
                    <span className="truncate text-muted-foreground">{name}</span>
                    <span className={pct >= 0 ? "font-medium text-green-600" : "font-medium text-red-600"}>
                      {pct >= 0 ? "+" : ""}
                      {pct.toFixed(2)}%
                    </span>
                  </div>
                ))}
              </div>
            </>
          )}

          {data.headlines.length > 0 && (
            <>
              <Separator />
              <div>
                <p className="mb-2 text-xs font-medium text-muted-foreground">
                  Headlines ({data.headline_count})
                </p>
                <ul className="space-y-1">
                  {data.headlines.map((h, i) => (
                    <li key={i} className="line-clamp-1 text-sm">
                      • {h}
                    </li>
                  ))}
                </ul>
              </div>
            </>
          )}
        </div>
      )}
    </Panel>
  );
}
