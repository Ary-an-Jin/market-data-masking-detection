"""
diagnose_leverage_effect.py
===========================
Tests whether real intraday NSE 1-minute data shows the standard negative
leverage effect (crashes -> higher future volatility) or a reversed positive
effect (rallies -> higher future volatility).

This is READ-ONLY. It does not touch your store, watermarks, or any config.
It reads directly from the parquet partitions using pathlib + pandas only.

Run from your project root with venv active:
  python diagnose_leverage_effect.py

Output: per-symbol table + overall conclusion for the paper.
"""
from __future__ import annotations

import sys
from math import gcd
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURE: point this at your point_in_time 1-min store
# ---------------------------------------------------------------------------
BASE_DIR   = Path(r"C:\Users\dell\OneDrive\Emailattachments\Predicitvemodel\Financialmodel\data\clean\point_in_time\market_intraday")
TIMEFRAME  = "1m"
MAX_DAYS   = 60       # most recent trading days per symbol (fast but sufficient)
MIN_ROWS   = 1000     # skip symbol if fewer rows (not enough for signal)
LAGS       = [1,2,3,4,5]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_symbols(base: Path) -> list[str]:
    """Return all symbol folder names found in the store."""
    return sorted(
        p.name.replace("symbol=", "")
        for p in base.iterdir()
        if p.is_dir() and p.name.startswith("symbol=")
    )


def _load_symbol(base: Path, symbol: str, timeframe: str, max_days: int) -> pd.DataFrame:
    """
    Load the most recent `max_days` trading days for one symbol.
    Returns a DataFrame with a DatetimeIndex and at least a 'close' column.
    """
    safe = symbol.replace("|", "_").replace("^", "").replace("/", "_")
    tf_dir = base / f"symbol={safe}" / f"timeframe={timeframe}"
    if not tf_dir.exists():
        return pd.DataFrame()

    # Sort date partitions descending, take the most recent max_days
    date_dirs = sorted(tf_dir.iterdir(), key=lambda p: p.name, reverse=True)
    date_dirs = [d for d in date_dirs if d.is_dir()][:max_days]

    parts = []
    for d in date_dirs:
        pq = d / "data.parquet"
        if pq.exists():
            try:
                parts.append(pd.read_parquet(pq))
            except Exception:
                pass

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts).sort_index()
    # Normalise index to DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ("datetime", "timestamp"):
            if col in df.columns:
                df = df.set_index(col)
                break
    return df


def _find_close_col(df: pd.DataFrame) -> str | None:
    """Find the close price column regardless of case."""
    for c in df.columns:
        if c.lower() == "close":
            return c
    return None


def _lagged_leverage(returns: pd.Series) -> tuple[float, float]:
    """
    Compute lagged leverage cross-correlation.
    L(tau) = corr(r_t, r_{t+tau}^2) for tau in LAGS.
    Returns (L(1), cumulative_L(1..5)).
    """
    r_sq = returns ** 2
    L_vals = []
    for tau in LAGS:
        r_past   = returns.iloc[:-tau].reset_index(drop=True)
        rsq_fut  = r_sq.iloc[tau:].reset_index(drop=True)
        if len(r_past) < 200:
            break
        c = float(r_past.corr(rsq_fut))
        if not np.isnan(c):
            L_vals.append(c)
    if not L_vals:
        return float("nan"), float("nan")
    return L_vals[0], sum(L_vals)


def _engle_ng_t(returns: pd.Series) -> float:
    """
    Engle-Ng sign bias test.
    OLS: r_t^2 = a0 + a1 * S^-_{t-1} + eps
    Returns t-statistic for a1.
    Positive t: negative past shock -> higher current vol (normal western).
    Negative t: positive past shock -> higher current vol (Indian-like / reversed).
    """
    r = returns.iloc[1:].reset_index(drop=True)
    s = (returns.iloc[:-1] < 0).astype(float).reset_index(drop=True)
    y = (r ** 2).values
    X = np.column_stack([np.ones(len(s)), s.values])
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        beta    = XtX_inv @ (X.T @ y)
        resid   = y - X @ beta
        mse     = np.sum(resid**2) / max(len(y)-2, 1)
        se      = np.sqrt(np.maximum(mse * np.diag(XtX_inv), 0.0))
        return float(beta[1] / se[1]) if se[1] > 1e-12 else float("nan")
    except Exception:
        return float("nan")


def _semivar_ratio(returns: pd.Series) -> float:
    """
    Downside semi-variance / upside semi-variance.
    > 1: losses more volatile than gains (normal western equity).
    < 1: gains more volatile (Indian-like / reversed).
    """
    down = returns[returns < 0]
    up   = returns[returns > 0]
    if len(down) < 20 or len(up) < 20:
        return float("nan")
    d_sv = float((down**2).mean())
    u_sv = float((up**2).mean())
    return d_sv / u_sv if u_sv > 1e-12 else float("nan")


def _gcd_tick(returns: pd.Series, prices: pd.Series) -> float | None:
    """Estimate tick size via GCD of non-zero price movements."""
    deltas  = prices.diff().abs().dropna()
    nonzero = deltas[deltas > 0]
    if len(nonzero) < 10:
        return None
    scale  = 100_000
    scaled = np.round(nonzero.values[:300] * scale).astype(np.int64)
    scaled = scaled[scaled > 0]
    if len(scaled) == 0:
        return None
    try:
        g = int(reduce(gcd, scaled.tolist()))
        return round(g / scale, 6) if g >= 10 else round(float(nonzero.min()), 6)
    except Exception:
        return round(float(nonzero.min()), 6)


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def run():
    print()
    print("=" * 72)
    print("  NSE INTRADAY LEVERAGE EFFECT DIAGNOSTIC")
    print(f"  Store : {BASE_DIR}")
    print(f"  TF    : {TIMEFRAME}  |  Days loaded: last {MAX_DAYS} per symbol")
    print("=" * 72)

    if not BASE_DIR.exists():
        print(f"\n  ERROR: Store path not found:\n  {BASE_DIR}")
        print("  Update BASE_DIR at the top of this script and re-run.")
        sys.exit(1)

    symbols = _find_symbols(BASE_DIR)
    if not symbols:
        print("\n  ERROR: No symbol=... folders found. Check BASE_DIR.")
        sys.exit(1)

    print(f"\n  Found {len(symbols)} symbol(s). Processing...\n")

    # Table header
    hdr = f"{'Symbol':<20} {'Rows':>7} {'Tick':>8} {'L(1)':>8} {'cumL':>8} {'EngNg-t':>9} {'SemiVar':>8}  Direction"
    print(hdr)
    print("-" * len(hdr))

    results = []

    for sym in symbols:
        df = _load_symbol(BASE_DIR, sym, TIMEFRAME, MAX_DAYS)
        if df.empty:
            print(f"  {'(no data)':<20}  {sym}")
            continue

        close_col = _find_close_col(df)
        if close_col is None:
            print(f"  {sym:<20}  (no close column)")
            continue

        prices  = df[close_col].dropna()
        if len(prices) < MIN_ROWS:
            print(f"  {sym:<20} {len(prices):>7}  (too few rows, skipping)")
            continue

        returns = prices.pct_change().dropna()

        L1, cumL   = _lagged_leverage(returns)
        t_stat     = _engle_ng_t(returns)
        sv_ratio   = _semivar_ratio(returns)
        tick       = _gcd_tick(returns, prices)
        n          = len(prices)

        # Determine direction from the two primary signals
        # Normal western: L1 < 0, cumL < 0, t > 0
        # Reversed Indian: L1 > 0, cumL > 0, t < 0
        signals_normal   = sum([
            1 if not np.isnan(cumL)    and cumL   < -0.01 else 0,
            1 if not np.isnan(t_stat)  and t_stat > 1.5   else 0,
            1 if not np.isnan(sv_ratio)and sv_ratio > 1.05 else 0,
        ])
        signals_reversed = sum([
            1 if not np.isnan(cumL)    and cumL   > 0.01  else 0,
            1 if not np.isnan(t_stat)  and t_stat < -1.5  else 0,
            1 if not np.isnan(sv_ratio)and sv_ratio < 0.95 else 0,
        ])

        if signals_normal >= 2:
            direction = "NORMAL (-)"
        elif signals_reversed >= 2:
            direction = "REVERSED (+)"
        else:
            direction = "weak / mixed"

        tick_str = f"{tick:.4f}" if tick else "  ?"
        L1_str   = f"{L1:.4f}"   if not np.isnan(L1)     else "   nan"
        cum_str  = f"{cumL:.4f}" if not np.isnan(cumL)    else "   nan"
        t_str    = f"{t_stat:.2f}" if not np.isnan(t_stat) else " nan"
        sv_str   = f"{sv_ratio:.4f}" if not np.isnan(sv_ratio) else "   nan"

        print(f"  {sym:<20} {n:>7} {tick_str:>8} {L1_str:>8} {cum_str:>8} {t_str:>9} {sv_str:>8}  {direction}")

        results.append({
            "symbol"   : sym,
            "n"        : n,
            "L1"       : L1,
            "cumL"     : cumL,
            "t_stat"   : t_stat,
            "sv_ratio" : sv_ratio,
            "direction": direction,
        })

    if not results:
        print("\n  No symbols processed successfully.")
        return

    # Summary
    print()
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    valid  = [r for r in results if r["direction"] != "weak / mixed"]
    normal = [r for r in valid if r["direction"] == "NORMAL (-)"]
    revsd  = [r for r in valid if r["direction"] == "REVERSED (+)"]
    mixed  = [r for r in results if r["direction"] == "weak / mixed"]

    print(f"\n  Total symbols analysed : {len(results)}")
    print(f"  Normal leverage (-)    : {len(normal)}  {[r['symbol'][:12] for r in normal]}")
    print(f"  Reversed leverage (+)  : {len(revsd)}  {[r['symbol'][:12] for r in revsd]}")
    print(f"  Weak / mixed           : {len(mixed)}")

    # Signal averages (excluding nan)
    def avg(key):
        vals = [r[key] for r in results if not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    print(f"\n  Mean L(1)       : {avg('L1'):.5f}  (negative = normal western leverage)")
    print(f"  Mean cumL(1..5) : {avg('cumL'):.5f}  (negative = normal western leverage)")
    print(f"  Mean EngNg-t    : {avg('t_stat'):.3f}   (positive = normal western leverage)")
    print(f"  Mean SemiVar    : {avg('sv_ratio'):.4f}  (> 1 = normal western leverage)")

    print()
    print("  INTERPRETATION FOR THE PAPER:")
    if len(normal) > len(revsd):
        print(f"  NSE intraday 1-min data shows NORMAL negative leverage in")
        print(f"  {len(normal)}/{len(results)} symbols tested.")
        print(f"  The 'reversed leverage' claim does NOT hold on your intraday data.")
        print(f"  => Reframe: leverage effect is PRESENT but WEAKER than western")
        print(f"     equity. Calibration still matters (magnitude, not sign).")
    elif len(revsd) > len(normal):
        print(f"  NSE intraday 1-min data shows REVERSED leverage in")
        print(f"  {len(revsd)}/{len(results)} symbols tested.")
        print(f"  => Original claim HOLDS on intraday data. Paper proceeds as planned.")
    else:
        print(f"  Mixed result -- {len(normal)} normal, {len(revsd)} reversed, {len(mixed)} weak.")
        print(f"  => Signal is not consistent enough to make a strong directional claim.")
        print(f"     Reframe around magnitude/calibration rather than sign direction.")

    print()
    print("  Paste the full output above back to confirm results.")
    print()


if __name__ == "__main__":
    run()
