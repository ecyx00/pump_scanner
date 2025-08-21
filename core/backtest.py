from dataclasses import dataclass
from typing import Dict, Sequence
import pandas as pd
import numpy as np
import numba


@dataclass
class BacktestConfig:
    horizons: Sequence[int] = (12, 48, 144, 288, 576)  # 5m bars: 1h, 4h, 12h, 24h, 48h
    mfe_mae_horizon: int = 576  # 48h in 5m bars


@numba.jit(nopython=True)
def _calculate_ttp_numba(arr: np.ndarray, fwd_max: np.ndarray, window: int) -> np.ndarray:
    """Numba ile JIT derlenmiş TTP hesaplama - hem bellek verimli hem de hızlı."""
    n = len(arr)
    ttp = np.full(n, np.nan)
    
    for i in range(n):
        if np.isnan(fwd_max[i]):
            continue
        
        limit = min(window + 1, n - i)
        for k in range(1, limit):
            if np.isclose(arr[i + k], fwd_max[i]):
                ttp[i] = k
                break
    
    return ttp


def _forward_returns_series(close: pd.Series, next_open: pd.Series, horizons: Sequence[int]) -> Dict[int, pd.Series]:
    returns: Dict[int, pd.Series] = {}
    for h in horizons:
        future_close = close.shift(-h)
        returns[h] = (future_close / next_open) - 1.0
    return returns


def _forward_extremes(high: pd.Series, low: pd.Series, window: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bellek verimli forward extremes hesaplama - matris tabanlı yaklaşım yerine döngü kullanır."""
    # Reverse time to use rolling window as forward-looking; exclude current bar via shift(1)
    high_rev = high.iloc[::-1]
    low_rev = low.iloc[::-1]

    fwd_max_high_rev = high_rev.rolling(window=window, min_periods=1).max().shift(1)
    fwd_min_low_rev = low_rev.rolling(window=window, min_periods=1).min().shift(1)

    fwd_max_high = fwd_max_high_rev.iloc[::-1]
    fwd_min_low = fwd_min_low_rev.iloc[::-1]

    # Numba ile optimize edilmiş TTP hesaplama - hem bellek verimli hem de hızlı
    ttp_values = _calculate_ttp_numba(high.to_numpy(), fwd_max_high.to_numpy(), window)
    ttp_series = pd.Series(ttp_values, index=high.index)
    
    return fwd_max_high, fwd_min_low, ttp_series


def compute_event_metrics(ohlc: pd.DataFrame, events: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """Compute per-event metrics without per-event loops.

    ohlc columns: ['ts','open','high','low','close'] for a single symbol, 5m bars, ts increasing.
    events columns: ['event_ts','entry_price_ref'] where entry_price_ref = next open after event bar.
    """
    ohlc = ohlc.sort_values("ts").reset_index(drop=True)
    ohlc["idx"] = np.arange(len(ohlc))

    # Map event timestamps to ohlc index
    idx_map = ohlc.set_index("ts")["idx"]
    ev = events.copy()
    ev["idx"] = ev["event_ts"].map(idx_map)
    ev = ev.dropna(subset=["idx"]).astype({"idx": int})

    # Next open and forward extremes
    next_open = ohlc["open"].shift(-1)
    fwd_max_high, fwd_min_low, ttp_series = _forward_extremes(ohlc["high"], ohlc["low"], config.mfe_mae_horizon)

    # Forward returns per horizon relative to next open (reference price)
    fwd_returns = _forward_returns_series(ohlc["close"], next_open, config.horizons)

    # Select metrics at event indices
    ev_indices = ev["idx"].to_numpy()
    pref = next_open.iloc[ev_indices].to_numpy()

    mfe = (fwd_max_high.iloc[ev_indices].to_numpy() / pref) - 1.0
    mae = (fwd_min_low.iloc[ev_indices].to_numpy() / pref) - 1.0
    ttp = ttp_series.iloc[ev_indices].to_numpy()

    out = pd.DataFrame({
        "event_ts": ev["event_ts"].values,
        "entry_price_ref": pref,
        "mfe_48h": mfe,
        "mae_48h": mae,
        "ttp": ttp,
    })

    for h, series in fwd_returns.items():
        out[f"R_{h}"] = series.iloc[ev_indices].to_numpy()

    # Drop events where entry reference is NaN (no next bar)
    out = out.dropna(subset=["entry_price_ref"])  # aligns with calculate.py behavior
    return out
