"use client";

import { api } from "@/lib/api";
import { useApi } from "@/lib/use-api";
import { Badge } from "@/components/ui/badge";
import { Panel } from "./panel";

function Dot({ on, label }: { on: boolean; label: string }) {
  return (
    <div className="flex items-center gap-2 text-sm">
      <span
        className={`inline-block h-2 w-2 rounded-full ${on ? "bg-green-500" : "bg-muted-foreground/40"}`}
      />
      <span className={on ? "" : "text-muted-foreground"}>{label}</span>
    </div>
  );
}

export function HealthCard() {
  const { data, error, loading, refetch } = useApi(api.health);
  return (
    <Panel title="System health" loading={loading} error={error} onRefresh={refetch}>
      {data && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Badge variant={data.status === "ok" ? "default" : "destructive"}>{data.status}</Badge>
            <Badge variant="secondary">mode: {data.mode}</Badge>
          </div>
          <div className="grid grid-cols-2 gap-x-4 gap-y-2">
            <Dot on={data.kite_configured} label="Kite" />
            <Dot on={data.telegram_configured} label="Telegram" />
            <Dot
              on={data.llm_providers_configured.length > 0}
              label={`LLM (${data.llm_providers_configured.length})`}
            />
          </div>
          {data.llm_providers_configured.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {data.llm_providers_configured.map((p) => (
                <Badge key={p} variant="outline" className="text-xs">
                  {p}
                </Badge>
              ))}
            </div>
          )}
        </div>
      )}
    </Panel>
  );
}
