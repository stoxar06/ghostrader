import { HealthCard } from "@/components/dashboard/health-card";
import { ConfigCard } from "@/components/dashboard/config-card";
import { BriefingCard } from "@/components/dashboard/briefing-card";
import { PriceCard } from "@/components/dashboard/price-card";
import { PnlCard } from "@/components/dashboard/pnl-card";
import { TradesCard } from "@/components/dashboard/trades-card";

export default function Home() {
  return (
    <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6">
      <header className="mb-6 flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Ghostrader</h1>
          <p className="text-sm text-muted-foreground">
            Read-only analysis dashboard · never trades
          </p>
        </div>
        <span className="rounded-md border px-2 py-1 text-xs text-muted-foreground">
          API: <code>/api</code> → Flask :5000
        </span>
      </header>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <HealthCard />
        <ConfigCard />
        <BriefingCard />
        <PriceCard />
        <PnlCard />
        <TradesCard />
      </div>
    </main>
  );
}
