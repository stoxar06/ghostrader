"""SIP & buy-and-hold return analytics.

- xirr: money-weighted return for irregular/periodic cashflows (bisection).
- sip: invest a fixed amount on the first trading day of each month -> XIRR.
- lumpsum: invest once at the start -> total return + CAGR.
"""
from __future__ import annotations

from datetime import date

import pandas as pd


def xirr(cashflows: list[tuple[date, float]]) -> float:
    """Money-weighted annual return. cashflows: (date, amount); outflows negative.

    Returns NaN if it can't bracket a root.
    """
    if len(cashflows) < 2:
        return float("nan")
    t0 = cashflows[0][0]

    def npv(rate: float) -> float:
        return sum(cf / ((1 + rate) ** ((d - t0).days / 365.0)) for d, cf in cashflows)

    lo, hi = -0.9999, 10.0
    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo * f_hi > 0:  # can't bracket a sign change
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if abs(f_mid) < 1e-7:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def _first_of_each_month(prices: pd.Series) -> pd.Series:
    px = prices.dropna()
    return px.groupby([px.index.year, px.index.month]).head(1)


def sip(prices: pd.Series, monthly_amount: float = 10_000.0) -> dict:
    """Simulate a monthly SIP and return invested/value/abs return/XIRR."""
    px = prices.dropna()
    if len(px) < 2:
        return {"invested": 0.0, "final_value": 0.0, "abs_return_pct": 0.0,
                "xirr_pct": float("nan"), "months": 0}

    units = invested = 0.0
    cashflows: list[tuple[date, float]] = []
    for ts, price in _first_of_each_month(px).items():
        if price <= 0:
            continue
        units += monthly_amount / price
        invested += monthly_amount
        cashflows.append((ts.date(), -monthly_amount))

    final_value = units * float(px.iloc[-1])
    cashflows.append((px.index[-1].date(), final_value))
    r = xirr(cashflows)
    return {
        "invested": invested,
        "final_value": final_value,
        "abs_return_pct": (final_value / invested - 1) * 100 if invested else 0.0,
        "xirr_pct": r * 100 if r == r else float("nan"),
        "months": len(cashflows) - 1,
    }


def lumpsum(prices: pd.Series) -> dict:
    """Invest once at the start; report total return + CAGR."""
    px = prices.dropna()
    if len(px) < 2:
        return {"total_return_pct": 0.0, "cagr_pct": 0.0, "years": 0.0}
    growth = float(px.iloc[-1] / px.iloc[0])
    years = (px.index[-1] - px.index[0]).days / 365.0
    cagr = growth ** (1 / years) - 1 if years > 0 and growth > 0 else float("nan")
    return {
        "total_return_pct": (growth - 1) * 100,
        "cagr_pct": cagr * 100 if cagr == cagr else 0.0,
        "years": round(years, 1),
    }


def _main() -> None:  # pragma: no cover - CLI, needs network
    from rich.console import Console
    from rich.table import Table

    from src.backtest.momentum import DEFAULT_UNIVERSE
    from src.data.historical import HistoricalData

    hist = HistoricalData(cache_dir="data/cache")
    rows = []
    port_cashflows: dict = {}
    monthly = 10_000.0
    per_name = monthly / len(DEFAULT_UNIVERSE)
    port_units: dict[str, float] = {}
    last_price: dict[str, float] = {}

    for sym in DEFAULT_UNIVERSE:
        try:
            px = hist.get(sym, "day")["close"].dropna()
        except Exception:  # noqa: BLE001
            continue
        s = sip(px, monthly)
        lp = lumpsum(px)
        rows.append((sym, s["xirr_pct"], lp["cagr_pct"], lp["years"]))
        # accumulate equal-weight portfolio SIP
        units = 0.0
        for ts, price in _first_of_each_month(px).items():
            if price > 0:
                units += per_name / price
                port_cashflows[ts.date()] = port_cashflows.get(ts.date(), 0.0) - per_name
        port_units[sym] = units
        last_price[sym] = float(px.iloc[-1])

    con = Console()
    table = Table(title="SIP (XIRR) & lumpsum (CAGR) per name — what actually works")
    for c in ("symbol", "SIP XIRR%", "lumpsum CAGR%", "years"):
        table.add_column(c)
    for sym, x, c, y in sorted(rows, key=lambda r: (r[1] if r[1] == r[1] else -999), reverse=True):
        table.add_row(sym, f"{x:.1f}", f"{c:.1f}", f"{y}")
    con.print(table)

    # Realistic, survivorship-free benchmark: Nifty 50 index SIP.
    try:
        nifty = hist.get("^NSEI", "day")["close"]
        ns = sip(nifty, monthly)
        con.print(f"[bold]NIFTY 50 index SIP (realistic, no survivorship bias)[/bold]: "
                  f"XIRR {ns['xirr_pct']:.1f}%  "
                  f"(₹{ns['invested']:,.0f} -> ₹{ns['final_value']:,.0f} over {ns['months']} months)")
    except Exception as exc:  # noqa: BLE001
        con.print(f"[yellow]Nifty benchmark unavailable: {exc}[/yellow]")

    if port_cashflows:
        final = sum(port_units[s] * last_price[s] for s in port_units)
        cfs = sorted(port_cashflows.items())
        last_day = max(last_price_day for last_price_day in [k for k, _ in cfs]) if cfs else None
        cfs.append((max(k for k, _ in cfs), final))  # inflow on last contribution date (approx)
        invested = -sum(v for _, v in cfs if v < 0)
        r = xirr(cfs)
        con.print(f"[bold]Equal-weight portfolio SIP[/bold]: invested ₹{invested:,.0f} -> "
                  f"₹{final:,.0f}  (XIRR {r*100:.1f}%)" if r == r else
                  f"[bold]Equal-weight portfolio SIP[/bold]: invested ₹{invested:,.0f} -> ₹{final:,.0f}")


if __name__ == "__main__":  # pragma: no cover
    _main()
