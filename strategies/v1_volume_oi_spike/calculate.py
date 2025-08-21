import os
from pathlib import Path
import pandas as pd
import numpy as np
import numba
from typing import Tuple
from datetime import datetime, timezone, timedelta
from core import data as core_data
from .config import Config


@numba.jit(nopython=True)
def _apply_cooldown_numba(triggers: np.ndarray, cooldown: int) -> np.ndarray:
    """JIT-compiled cooldown logic for boolean trigger array."""
    result = triggers.copy()
    last_idx = -1
    
    for i in range(len(triggers)):
        if last_idx >= 0 and i - last_idx <= cooldown:
            result[i] = False
        if triggers[i]:
            last_idx = i
    
    return result


def zscore_grouped(series: pd.Series, window: int) -> pd.Series:
    """Compute z-score within each group using rolling operations."""
    rolling_mean = series.groupby(level=0).rolling(window=window, min_periods=window//4).mean().reset_index(0, drop=True)
    rolling_std = series.groupby(level=0).rolling(window=window, min_periods=window//4).std(ddof=0).reset_index(0, drop=True)
    return (series - rolling_mean) / rolling_std.replace(0.0, np.nan)


def price_thrust_grouped(df: pd.DataFrame, lookback: int) -> pd.Series:
    """Compute price thrust within each group using rolling operations."""
    highest = df.groupby('symbol_id')['high'].rolling(window=lookback, min_periods=1).max().reset_index(0, drop=True)
    lowest = df.groupby('symbol_id')['low'].rolling(window=lookback, min_periods=1).min().reset_index(0, drop=True)
    denom = (highest - lowest).replace(0.0, np.nan)
    return (df['close'] - lowest) / denom


def build_score_grouped(df: pd.DataFrame, cfg: Config) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Build anomaly score using vectorized operations on grouped data."""
    vol_mix = 0.6 * df["spot_base_vol"].astype(float) + 0.4 * df["perp_base_vol"].astype(float)
    z_vol = zscore_grouped(vol_mix, cfg.nz)

    oi_delta = df.groupby('symbol_id')['oi'].diff()
    z_oi = zscore_grouped(oi_delta, cfg.nz)

    p_thrust = price_thrust_grouped(df, cfg.np_)

    score = cfg.w_vol * z_vol + cfg.w_oi * z_oi + cfg.w_price * p_thrust
    return score, z_vol, z_oi


def rising_edge_events_grouped(score: pd.Series, threshold: pd.Series, cooldown_bars: int) -> pd.Series:
    """Detect rising edge events within each group with cooldown enforcement."""
    above = score > threshold
    rising = above & (~above.groupby(level=0).shift(1, fill_value=False))
    
    if cooldown_bars <= 0:
        return rising
    
    # Apply cooldown using JIT-compiled numba function
    def apply_cooldown_wrapper(group):
        triggers_np = group.to_numpy()
        result_np = _apply_cooldown_numba(triggers_np, cooldown_bars)
        return pd.Series(result_np, index=group.index)
    
    return rising.groupby(level=0).apply(apply_cooldown_wrapper).reset_index(0, drop=True)


async def load_data(client: core_data.DataClient, symbol_ids: list[int], start_ts: datetime, end_ts: datetime) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Veri çekme işlemini gerçekleştirir."""
    print(f"Veri çekiliyor: {len(symbol_ids)} sembol için...")
    
    perp = await client.fetch_perp_klines_5m(symbol_ids, start_ts, end_ts)
    spot = await client.fetch_spot_klines_5m(symbol_ids, start_ts, end_ts)
    oi = await client.fetch_oi_5m(symbol_ids, start_ts, end_ts)
    
    print(f"Veri çekildi: {len(perp)} perp, {len(spot)} spot, {len(oi)} OI bar")
    return perp, spot, oi


def prepare_features(perp: pd.DataFrame, spot: pd.DataFrame, oi: pd.DataFrame) -> pd.DataFrame:
    """Verileri birleştirir ve özellikleri hazırlar."""
    # 1. Perp verisini hazırla
    if 'base_vol' not in perp.columns:
        raise ValueError("Gerekli 'base_vol' kolonu perp DataFrame'inde bulunamadı.")
    perp_selected = perp.rename(columns={"base_vol": "perp_base_vol"})
    perp_selected = perp_selected[["symbol_id", "ts", "open", "high", "low", "close", "perp_base_vol"]]

    # 2. Spot verisini hazırla
    if 'base_vol' not in spot.columns:
        spot['spot_base_vol'] = np.nan
    else:
        spot = spot.rename(columns={"base_vol": "spot_base_vol"})
    spot_selected = spot[["symbol_id", "ts", "spot_base_vol"]]

    # 3. OI verisini hazırla
    if 'oi' not in oi.columns:
        raise ValueError("Gerekli 'oi' kolonu oi DataFrame'inde bulunamadı.")
    oi_selected = oi[["symbol_id", "ts", "oi"]]

    # 4. Tüm veri kaynaklarını birleştir
    merged = pd.merge(perp_selected, spot_selected, on=["symbol_id", "ts"], how="left")
    merged = pd.merge(merged, oi_selected, on=["symbol_id", "ts"], how="left")

    # spot_base_vol'deki boşlukları 0 ile doldur
    merged['spot_base_vol'] = merged['spot_base_vol'].fillna(0)
    
    # Sort by symbol_id and timestamp for groupby operations
    merged = merged.sort_values(["symbol_id", "ts"]).reset_index(drop=True)
    
    # Set symbol_id as index for groupby operations
    merged = merged.set_index("symbol_id")
    
    return merged


def compute_signals(features: pd.DataFrame, cfg: Config) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Skorları ve tetikleyicileri hesaplar."""
    # Compute scores and thresholds using vectorized groupby operations
    score, z_vol, z_oi = build_score_grouped(features, cfg)
    
    # Compute dynamic threshold per symbol using groupby
    threshold = score.groupby(level=0).shift(1).rolling(
        window=cfg.nz, min_periods=max(10, cfg.nz//10)
    ).quantile(cfg.trigger_percentile/100.0).fillna(method="bfill").fillna(
        score.groupby(level=0).rolling(window=cfg.nz, min_periods=1).median()
    )
    
    # Detect events using vectorized operations
    triggers = rising_edge_events_grouped(score, threshold, cfg.cooldown_bars)
    
    return score, z_vol, z_oi, threshold, triggers


def generate_events(signals: tuple, features: pd.DataFrame, name_to_id: dict, cfg: Config) -> pd.DataFrame:
    """Events DataFrame'ini oluşturur."""
    score, z_vol, z_oi, threshold, triggers = signals
    
    # Get next bar open as entry reference price
    next_open = features.groupby(level=0)['open'].shift(-1)
    
    # Create events DataFrame - verimli ters haritalama kullan
    id_to_name = {v: k for k, v in name_to_id.items()}
    events = pd.DataFrame({
        "symbol": features.index.map(id_to_name),
        "event_ts": features.loc[triggers, "ts"],
        "entry_price_ref": next_open[triggers],
        "score": score[triggers],
        "score_threshold": threshold[triggers],
        "z_vol": z_vol[triggers],
        "z_oi": z_oi[triggers],
        "price_thrust": price_thrust_grouped(features, cfg.np_)[triggers],
        "params_version": "v2_default",
    }).reset_index(drop=True)
    
    # Drop events where entry reference is NaN (no next bar)
    events = events.dropna(subset=["entry_price_ref"])
    
    return events


async def add_adv_data(events: pd.DataFrame, name_to_id: dict, client: core_data.DataClient) -> pd.DataFrame:
    """ADV verisini events DataFrame'ine ekler."""
    print("ADV hesaplanıyor...")
    
    # ADV verisini toplu olarak çek (verimli yöntem)
    event_symbol_ids = [name_to_id[s] for s in events["symbol"].unique() if s in name_to_id]
    if event_symbol_ids:
        adv_df = await client.fetch_average_daily_volume_bulk(event_symbol_ids, days=30)
        id_to_adv = {row['symbol_id']: row['adv'] for _, row in adv_df.iterrows()}
        
        # ADV'yi events DataFrame'ine ekle
        id_to_name = {v: k for k, v in name_to_id.items()}
        events['adv'] = events['symbol'].map(name_to_id).map(id_to_adv)
        events['adv'] = events['adv'].fillna(0.0)
        
        print(f"ADV hesaplaması tamamlandı: {len(id_to_adv)} sembol")
    else:
        # Hiç sembol yoksa boş ADV sütunu ekle
        events['adv'] = 0.0
        print("ADV hesaplaması: Hiç sembol bulunamadı")
    
    return events


def save_events(events: pd.DataFrame, package_name: str) -> Path:
    """Events DataFrame'ini parquet dosyasına kaydeder."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = Path("out") / package_name / today
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "events.parquet"
    events.to_parquet(out_path, index=False)
    print(f"Wrote {len(events)} events -> {out_path}")
    return out_path


async def main_async():
    """Ana orkestratör fonksiyonu - küçük fonksiyonları sırayla çağırır."""
    cfg_path = Path(__file__).with_name("config.yml")
    cfg = Config.load(str(cfg_path))

    symbols_df, name_to_id = await core_data.load_enabled_symbols_async(cfg.symbols_csv)
    selected = symbols_df[symbols_df["enabled"] == 1]["symbol"].tolist()
    symbol_ids = [name_to_id[s] for s in selected]

    # Dinamik zaman aralığı: son 30 gün
    end_ts = datetime.now(timezone.utc)
    start_ts = end_ts - timedelta(days=30)
    
    print(f"Zaman aralığı: {start_ts.strftime('%Y-%m-%d %H:%M')} - {end_ts.strftime('%Y-%m-%d %H:%M')}")
    
    # DB-only: will raise if DB not configured or unreachable
    client = core_data.get_db_client()
    if not client:
        raise RuntimeError("Veritabanı bağlantısı kurulamadı")
    
    # 1. Veri yükle
    perp, spot, oi = await load_data(client, symbol_ids, start_ts, end_ts)
    
    # 2. Özellikleri hazırla
    features = prepare_features(perp, spot, oi)
    
    # 3. Sinyalleri hesapla
    signals = compute_signals(features, cfg)
    
    # 4. Events DataFrame'ini oluştur
    events = generate_events(signals, features, name_to_id, cfg)
    
    # 5. ADV verisini ekle
    events = await add_adv_data(events, name_to_id, client)
    
    # 6. Sonuçları kaydet
    save_events(events, Path(__package__).name)


def main():
    """Senkron wrapper - async main fonksiyonunu çalıştırır"""
    import asyncio
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
