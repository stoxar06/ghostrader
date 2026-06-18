// Typed client for the Ghostrader Flask API (proxied via next.config rewrites).
// Shapes mirror src/web/server.py + src/mcp/tools.py.

export interface Health {
  status: string;
  mode: string;
  llm_providers_configured: string[];
  kite_configured: boolean;
  telegram_configured: boolean;
}

export interface Config {
  mode: string;
  instruments: Record<string, unknown> & { base_timeframe?: string };
  risk: Record<string, unknown>;
  llm: { provider_order?: string[]; daily_paid_cap_inr?: number };
  research_universe: string[];
}

export interface Regime {
  regime: string;
  bias: string;
  favored_sectors: string[];
  avoid_sectors: string[];
  summary: string;
  source: string;
}

export interface Briefing {
  regime: Regime;
  cues: Record<string, number>;
  headline_count: number;
  headlines: string[];
}

export interface Prices {
  symbol: string;
  dates: string[];
  close: number[];
}

export interface DailyPnL {
  day: string;
  realized_pnl: number;
  trades: number;
  halted: boolean;
  mode: string;
}

export interface TradeRow {
  symbol: string;
  side: string;
  qty: number;
  entry: number;
  exit: number | null;
  pnl: number | null;
  reason: string;
  mode: string;
  day: string | null; // exit date (replay sessions are historical)
}

export interface EdgeHofRow {
  config: string;
  family: string;
  params: Record<string, unknown>;
  fitness_pp: number;
  horizon: number;
  oos_accuracy: number;
  oos_cond_edge_pp: number;
  is_cond_edge_pp: number;
  oos_signals: number;
}

export interface EdgeReport {
  edge_archive: {
    n_winners: number;
    n_real_edge: number;
    n_drift_riders: number;
    winners: { config: string; rule: string; horizon: number; oos_accuracy: number;
               oos_signals: number; oos_cond_edge_pp: number; label: string }[];
  } | null;
  auto_search: {
    generations: number;
    configs_tried: number;
    suspects: number;
    last_stop: string;
    hall_of_fame: EdgeHofRow[];
    verdict: { champion?: string; z?: number; z_expected_by_luck?: number;
               meaningful: boolean; note: string };
  } | null;
}

export interface HorizonRow {
  signals: number;
  win_rate: number;
  avg_ret_pct: number;
  total_pnl_pct: number;
  base_up_rate: number;
  edge_vs_base: number;
  profit_days?: number;
  loss_days?: number;
}

export interface HorizonDaily {
  days: { date: string; pnl_pct: number; trades: number }[];
  profit_days: number;
  loss_days: number;
  total_pnl_pct: number;
}

export interface HorizonReport {
  timeframe: string;
  symbols: number;
  cost_pct: number;
  horizons: Record<string, HorizonRow>;
  daily: Record<string, HorizonDaily>;
}

export interface NewsItem {
  title: string;
  link: string;
  source: string;
  published: string;
  published_ts: number;
}

export interface CompanyNewsItem extends NewsItem {
  symbol: string;
  company: string;
}

export interface IndicatorRow {
  indicator: string;
  info: string;
  signals: number;
  coverage: number;
  accuracy: number;
  base: number;
  edge_pp: number;
}

export interface VolumeRow {
  indicator: string;
  info: string;
  base_acc: number;
  vol_acc: number;
  delta_pp: number;
  vol_signals: number;
  vol_edge_pp: number;
}

export interface ConfluenceRow {
  rule: string;
  info: string;
  signals: number;
  coverage: number;
  accuracy: number;
  base: number;
  edge_pp: number;
}

export interface AccuracyReport {
  timeframe: string;
  horizon: number;
  symbols: number;
  base: number;
  base_up_rate: number;
  single: IndicatorRow[];
  volume: VolumeRow[];
  confluence: ConfluenceRow[];
}

export class ApiError extends Error {}

export async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(path, { signal, cache: "no-store" });
  const body = await res.json().catch(() => ({}));
  if (!res.ok || (body && typeof body === "object" && "error" in body)) {
    throw new ApiError((body as { error?: string })?.error ?? `Request failed (${res.status})`);
  }
  return body as T;
}

export const api = {
  health: (s?: AbortSignal) => getJson<Health>("/api/health", s),
  config: (s?: AbortSignal) => getJson<Config>("/api/config", s),
  briefing: (s?: AbortSignal) => getJson<Briefing>("/api/briefing", s),
  prices: (symbol: string, tf = "day", s?: AbortSignal) =>
    getJson<Prices>(`/api/prices?symbol=${encodeURIComponent(symbol)}&tf=${tf}`, s),
  dailyPnl: (s?: AbortSignal) => getJson<DailyPnL[]>("/api/daily_pnl", s),
  trades: (days?: number, s?: AbortSignal) =>
    getJson<TradeRow[]>(`/api/trades?limit=200${days ? `&days=${days}` : ""}`, s),
  edge: (s?: AbortSignal) => getJson<EdgeReport>("/api/edge", s),
  news: (limit = 40, s?: AbortSignal) => getJson<NewsItem[]>(`/api/news?limit=${limit}`, s),
  companyNews: (s?: AbortSignal) => getJson<CompanyNewsItem[]>("/api/company_news", s),
  indicatorAccuracy: (horizon = 5, s?: AbortSignal) =>
    getJson<AccuracyReport>(`/api/indicator_accuracy?horizon=${horizon}`, s),
  horizon: (maxDays = 12, s?: AbortSignal) =>
    getJson<HorizonReport>(`/api/horizon?max_days=${maxDays}`, s),
};

/** "3h ago" style relative time from a unix-seconds timestamp. */
export function timeAgo(ts: number): string {
  if (!ts) return "";
  const secs = Math.max(0, Math.floor(Date.now() / 1000 - ts));
  const units: [number, string][] = [
    [86400, "d"],
    [3600, "h"],
    [60, "m"],
  ];
  for (const [size, label] of units) {
    if (secs >= size) return `${Math.floor(secs / size)}${label} ago`;
  }
  return "just now";
}
