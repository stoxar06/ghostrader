"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import { useApi } from "@/lib/use-api";
import { Button } from "@/components/ui/button";
import { Panel } from "./panel";
import { LineChart } from "./line-chart";

const SYMBOLS = ["^NSEI", "^NSEBANK", "RELIANCE.NS", "TCS.NS", "INFY.NS"];

export function PriceCard() {
  const [symbol, setSymbol] = useState(SYMBOLS[0]);
  const { data, error, loading, refetch } = useApi((s) => api.prices(symbol, "day", s), [symbol]);

  return (
    <Panel title="Price history (daily)" loading={loading} error={error} onRefresh={refetch} className="lg:col-span-3">
      <div className="mb-3 flex flex-wrap gap-1">
        {SYMBOLS.map((s) => (
          <Button
            key={s}
            variant={s === symbol ? "default" : "outline"}
            size="sm"
            className="h-7 px-2 text-xs"
            onClick={() => setSymbol(s)}
          >
            {s}
          </Button>
        ))}
      </div>
      {data && <LineChart dates={data.dates} values={data.close} />}
    </Panel>
  );
}
