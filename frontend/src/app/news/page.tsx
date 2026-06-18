"use client";

import { useMemo, useState } from "react";
import { api } from "@/lib/api";
import { useApi } from "@/lib/use-api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { BreakingBanner } from "@/components/news/breaking-banner";
import { NewsList } from "@/components/news/news-list";

function Loading() {
  return (
    <div className="space-y-3">
      {Array.from({ length: 5 }).map((_, i) => (
        <div key={i} className="space-y-1">
          <Skeleton className="h-4 w-4/5" />
          <Skeleton className="h-3 w-1/3" />
        </div>
      ))}
    </div>
  );
}

export default function NewsPage() {
  const company = useApi(api.companyNews);
  const market = useApi((s) => api.news(40, s));
  const [symbol, setSymbol] = useState<string | null>(null);

  const symbols = useMemo(
    () => Array.from(new Set((company.data ?? []).map((it) => it.symbol))).sort(),
    [company.data]
  );
  const filtered = useMemo(
    () => (company.data ?? []).filter((it) => !symbol || it.symbol === symbol),
    [company.data, symbol]
  );

  return (
    <main className="mx-auto w-full max-w-7xl flex-1 space-y-6 px-4 py-6 sm:px-6">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">News</h1>
          <p className="text-sm text-muted-foreground">
            Breaking company news + market headlines · refreshed every 5 min
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            company.refetch();
            market.refetch();
          }}
          disabled={company.loading || market.loading}
        >
          Refresh all
        </Button>
      </div>

      {/* Breaking banner — company-specific */}
      {company.loading ? (
        <Skeleton className="h-14 w-full rounded-lg" />
      ) : company.data ? (
        <BreakingBanner items={company.data} />
      ) : null}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Company news */}
        <Card className="lg:col-span-2">
          <CardHeader className="space-y-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Company news ({filtered.length})
            </CardTitle>
            {symbols.length > 0 && (
              <div className="flex flex-wrap gap-1">
                <Button
                  variant={symbol === null ? "default" : "outline"}
                  size="sm"
                  className="h-7 px-2 text-xs"
                  onClick={() => setSymbol(null)}
                >
                  All
                </Button>
                {symbols.map((s) => (
                  <Button
                    key={s}
                    variant={symbol === s ? "default" : "outline"}
                    size="sm"
                    className="h-7 px-2 text-xs"
                    onClick={() => setSymbol(s)}
                  >
                    {s}
                  </Button>
                ))}
              </div>
            )}
          </CardHeader>
          <CardContent>
            {company.loading ? (
              <Loading />
            ) : company.error ? (
              <p className="text-sm text-destructive">{company.error}</p>
            ) : (
              <NewsList items={filtered} />
            )}
          </CardContent>
        </Card>

        {/* Market news */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Market headlines
            </CardTitle>
          </CardHeader>
          <CardContent>
            {market.loading ? (
              <Loading />
            ) : market.error ? (
              <p className="text-sm text-destructive">{market.error}</p>
            ) : (
              <NewsList items={market.data ?? []} />
            )}
          </CardContent>
        </Card>
      </div>
    </main>
  );
}
