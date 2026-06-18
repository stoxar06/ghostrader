"use client";

import { api } from "@/lib/api";
import { useApi } from "@/lib/use-api";
import { Badge } from "@/components/ui/badge";
import { Panel } from "./panel";

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-right font-medium">{value}</span>
    </div>
  );
}

export function ConfigCard() {
  const { data, error, loading, refetch } = useApi(api.config);
  return (
    <Panel title="Configuration" loading={loading} error={error} onRefresh={refetch}>
      {data && (
        <div className="space-y-2">
          <Row label="Mode" value={<Badge variant="secondary">{data.mode}</Badge>} />
          <Row label="Base timeframe" value={String(data.instruments?.base_timeframe ?? "—")} />
          <Row
            label="Daily paid LLM cap"
            value={data.llm?.daily_paid_cap_inr != null ? `₹${data.llm.daily_paid_cap_inr}` : "—"}
          />
          <Row
            label="LLM order"
            value={
              <span className="flex flex-wrap justify-end gap-1">
                {(data.llm?.provider_order ?? []).map((p) => (
                  <Badge key={p} variant="outline" className="text-xs">
                    {p}
                  </Badge>
                ))}
              </span>
            }
          />
          <Row label="Universe" value={`${data.research_universe?.length ?? 0} symbols`} />
        </div>
      )}
    </Panel>
  );
}
