from pathlib import Path
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
from core.backtest import BacktestConfig, compute_event_metrics
from core import data as core_data
from .config import Config


def kpi_summary(df: pd.DataFrame) -> dict:
    out = {}
    if df.empty:
        return out
    medians = {k: float(np.nanmedian(df[k])) for k in df.columns if k.startswith("R_")}
    out["medians"] = medians
    out["winrate_R_24h"] = float(np.mean(df.get("R_288", pd.Series(dtype=float)) > 0)) if "R_288" in df else np.nan
    out["median_mfe_48h"] = float(np.nanmedian(df.get("mfe_48h", pd.Series(dtype=float))))
    out["median_mae_48h"] = float(np.nanmedian(df.get("mae_48h", pd.Series(dtype=float))))
    out["num_events"] = int(len(df))
    return out


def render_section(title: str, metrics: pd.DataFrame) -> str:
    figs = []
    horizons = sorted([int(c.split("_")[1]) for c in metrics.columns if c.startswith("R_")])
    if horizons:
        med = [np.nanmedian(metrics[f"R_{h}"]) for h in horizons]
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=horizons, y=med, mode="lines+markers", name="Median R_h"))
        fig1.update_layout(title=f"{title} — İleriye Dönük Getiri (Median)", xaxis_title="Bar (5m)", yaxis_title="Return")
        figs.append(fig1)

    if not metrics.empty and {"mae_48h","mfe_48h"}.issubset(metrics.columns):
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=metrics["mae_48h"], y=metrics["mfe_48h"], mode="markers", name="Events"))
        fig2.update_layout(title=f"{title} — MFE vs MAE (48h)", xaxis_title="MAE", yaxis_title="MFE")
        figs.append(fig2)

    if "ttp" in metrics:
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=metrics["ttp"], nbinsx=50))
        fig3.update_layout(title=f"{title} — TTP Histogramı (bars)", xaxis_title="Bars", yaxis_title="Count")
        figs.append(fig3)

    html_parts = [f.to_html(full_html=False, include_plotlyjs=False) for f in figs]
    return "\n".join(html_parts) if html_parts else f"<h4>{title}</h4><p>No figures</p>"


def load_events(package_dir: Path) -> tuple[pd.DataFrame, Path]:
    """Events parquet dosyasını yükler ve çıktı dizinini hazırlar."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = Path("out") / package_dir.name / today
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "events.parquet"
    
    if not events_path.exists():
        raise FileNotFoundError(f"events.parquet not found at {events_path}. Run calculate step first.")

    events = pd.read_parquet(events_path)
    if events.empty:
        (out_dir / "summary.json").write_text(json.dumps({"num_events": 0}), encoding="utf-8")
        (out_dir / "report.html").write_text("<h3>No events</h3>", encoding="utf-8")
        return events, out_dir
    
    return events, out_dir


async def load_ohlc_data(events: pd.DataFrame, name_to_id: dict, start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
    """Tüm semboller için OHLC verisini tek seferde çeker."""
    all_symbol_ids = [name_to_id.get(s) for s in events["symbol"].unique() if name_to_id.get(s) is not None]
    
    print(f"Tüm semboller için OHLC verisini tek seferde çekiliyor...")
    client = core_data.get_db_client()
    if not client:
        raise RuntimeError("Veritabanı bağlantısı kurulamadı")
    
    all_ohlc_data = await client.fetch_perp_klines_5m(all_symbol_ids, start_ts, end_ts)
    print(f"Toplam {len(all_symbol_ids)} sembol için {len(all_ohlc_data)} OHLC bar yüklendi")
    
    return all_ohlc_data


def compute_metrics_per_symbol(events: pd.DataFrame, all_ohlc_data: pd.DataFrame, name_to_id: dict) -> dict:
    """Her sembol için metrikleri hesaplar."""
    all_ohlc_data_grouped = all_ohlc_data.groupby('symbol_id')
    metrics_per_symbol = {}
    
    for sym in sorted(events["symbol"].unique()):
        sym_id = name_to_id.get(sym)
        if sym_id is None:
            continue
        
        # Veritabanı sorgusu yerine bellekten filtreleme
        if sym_id not in all_ohlc_data_grouped.groups:
            print(f"UYARI: {sym} için OHLC verisi bulunamadı, atlanıyor")
            continue
        
        ohlc = all_ohlc_data_grouped.get_group(sym_id)
        ohlc = ohlc[["ts","open","high","low","close"]].sort_values("ts").reset_index(drop=True)
        ev = events[events.symbol == sym][["event_ts","entry_price_ref"]].reset_index(drop=True)
        m = compute_event_metrics(ohlc, ev, BacktestConfig())
        m["symbol"] = sym
        metrics_per_symbol[sym] = m
    
    return metrics_per_symbol


def extract_adv_data(events: pd.DataFrame) -> dict:
    """Events DataFrame'inden ADV verisini çıkarır."""
    print("ADV verisi events.parquet'ten okunuyor...")
    
    adv_map = {}
    for symbol_name in events["symbol"].unique():
        symbol_events = events[events["symbol"] == symbol_name]
        if not symbol_events.empty and "adv" in symbol_events.columns:
            # Her sembol için ortalama ADV (birden fazla olay varsa)
            avg_adv = symbol_events["adv"].mean()
            adv_map[symbol_name] = float(avg_adv)
        else:
            adv_map[symbol_name] = 0.0
    
    print(f"ADV verisi yüklendi: {len(adv_map)} sembol")
    return adv_map


def assign_tiers(metrics_per_symbol: dict, adv_map: dict) -> pd.DataFrame:
    """Metriklere tier bilgisi ekler."""
    def tier_for(adv: float) -> str:
        if adv > 50_000_000:
            return "Tier 1 (>50M)"
        if adv >= 10_000_000:
            return "Tier 2 (10M-50M)"
        return "Tier 3 (<10M)"

    metrics_all = pd.concat(metrics_per_symbol.values(), ignore_index=True) if metrics_per_symbol else pd.DataFrame()
    symbol_to_tier = {s: tier_for(adv_map.get(s, 0.0)) for s in metrics_per_symbol.keys()}
    metrics_all["tier"] = metrics_all["symbol"].map(symbol_to_tier)
    
    return metrics_all


def generate_report(metrics_all: pd.DataFrame, out_dir: Path) -> None:
    """HTML raporu ve JSON özeti oluşturur."""
    # Build sections
    sections_html = []
    # General section
    sections_html.append("<h3>Genel</h3>" + render_section("Genel", metrics_all))

    # Tiers
    for tier_name in ["Tier 1 (>50M)", "Tier 2 (10M-50M)", "Tier 3 (<10M)"]:
        tier_df = metrics_all[metrics_all["tier"] == tier_name]
        sections_html.append(f"<h3>{tier_name}</h3>" + render_section(tier_name, tier_df))

    # Write report with a single PlotlyJS include
    report_html = [
        "<html><head>",
        '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>',
        "</head><body>",
        *sections_html,
        "</body></html>",
    ]
    (out_dir / "report.html").write_text("\n".join(report_html), encoding="utf-8")

    # Summary per section
    summary = {
        "overall": kpi_summary(metrics_all),
        "tier1": kpi_summary(metrics_all[metrics_all["tier"] == "Tier 1 (>50M)"] ),
        "tier2": kpi_summary(metrics_all[metrics_all["tier"] == "Tier 2 (10M-50M)"] ),
        "tier3": kpi_summary(metrics_all[metrics_all["tier"] == "Tier 3 (<10M)"] ),
        "num_events": int(len(metrics_all)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote segmented report -> {out_dir / 'report.html'}")


async def main_async():
    """Ana orkestratör fonksiyonu - küçük fonksiyonları sırayla çağırır."""
    pkg_dir = Path(__file__).parent
    cfg = Config.load(str(pkg_dir / "config.yml"))

    # 1. Events dosyasını yükle
    events, out_dir = load_events(pkg_dir)
    if events.empty:
        return

    # 2. Dinamik zaman aralığı: son 30 gün
    end_ts = datetime.now(timezone.utc)
    start_ts = end_ts - timedelta(days=30)
    print(f"Zaman aralığı: {start_ts.strftime('%Y-%m-%d %H:%M')} - {end_ts.strftime('%Y-%m-%d %H:%M')}")
    
    # 3. Sembol bilgilerini yükle
    symbols_df, name_to_id = await core_data.load_enabled_symbols_async(cfg.symbols_csv)

    # 4. OHLC verisini yükle
    all_ohlc_data = await load_ohlc_data(events, name_to_id, start_ts, end_ts)

    # 5. Her sembol için metrikleri hesapla
    metrics_per_symbol = compute_metrics_per_symbol(events, all_ohlc_data, name_to_id)

    # 6. ADV verisini çıkar
    adv_map = extract_adv_data(events)

    # 7. Tier'ları ata
    metrics_all = assign_tiers(metrics_per_symbol, adv_map)

    # 8. Raporu oluştur
    generate_report(metrics_all, out_dir)


def main():
    """Senkron wrapper - async main fonksiyonunu çalıştırır"""
    import asyncio
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
