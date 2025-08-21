#!/usr/bin/env python3
"""
QuantZilla - Data Ingestor
Son 30 günlük kline ve OI verilerini Binance API'den çeker ve veritabanına kaydeder.
"""

import sys
import asyncio
import aiohttp
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, List
from aiohttp_retry import RetryClient, ExponentialRetry
from core import data as core_data


# ============================================================================
# PARSER KATMANI - Binance API'den veri çekme ve parse etme
# ============================================================================

async def fetch_and_parse_futures_klines(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str = "5m",
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: int = 1500
) -> pd.DataFrame:
    """
    Binance USDⓈ-M Futures Kline verisini çeker ve parse eder.
    
    Args:
        session: aiohttp session
        symbol: Trading pair (e.g., "BTCUSDT")
        interval: Kline interval (e.g., "5m")
        start_time: Start time in milliseconds (Unix timestamp)
        end_time: End time in milliseconds (Unix timestamp)
        limit: Number of klines, max 1500
    
    Returns:
        pd.DataFrame: Parsed DataFrame with required schema
    """
    # Binance Futures API endpoint - rate limiting uygulanır
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    
    if start_time is not None:
        params["startTime"] = start_time
    if end_time is not None:
        params["endTime"] = end_time
    
    async with session.get(url, params=params) as response:
        if response.status != 200:
            text = await response.text()
            raise Exception(f"API error: {response.status} - {text}")
        
        data = await response.json()
        if not data:
            return pd.DataFrame()
    
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'base_vol',
        'close_time', 'quote_vol', 'num_trades', 'taker_buy_base_vol',
        'taker_buy_quote_vol', 'ignore'
    ]
    df = pd.DataFrame(data, columns=columns)
    
    # Convert timestamp to datetime
    df['ts'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    
    # Select and rename columns as required
    df = df[['ts', 'open', 'high', 'low', 'close', 'base_vol', 'quote_vol', 'taker_buy_base_vol', 'taker_buy_quote_vol']]
    
    # Convert string fields to float
    float_cols = ['open', 'high', 'low', 'close', 'base_vol', 'quote_vol', 'taker_buy_base_vol', 'taker_buy_quote_vol']
    df[float_cols] = df[float_cols].astype(float)
    
    return df


async def fetch_and_parse_spot_klines(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str = "5m",
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: int = 1000
) -> pd.DataFrame:
    """
    Binance Spot Kline verisini çeker ve parse eder.
    
    Args:
        session: aiohttp session
        symbol: Trading pair (e.g., "BTCUSDT")
        interval: Kline interval (e.g., "5m")
        start_time: Start time in milliseconds (Unix timestamp)
        end_time: End time in milliseconds (Unix timestamp)
        limit: Number of klines, max 1000
    
    Returns:
        pd.DataFrame: Parsed DataFrame with required schema
    """
    # Binance Spot API endpoint - rate limiting uygulanır
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    
    if start_time is not None:
        params["startTime"] = start_time
    if end_time is not None:
        params["endTime"] = end_time
    
    async with session.get(url, params=params) as response:
        if response.status != 200:
            text = await response.text()
            raise Exception(f"API error: {response.status} - {text}")
        
        data = await response.json()
        if not data:
            return pd.DataFrame()
    
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'base_vol',
        'close_time', 'quote_vol', 'num_trades', 'taker_buy_base_vol',
        'taker_buy_quote_vol', 'ignore'
    ]
    df = pd.DataFrame(data, columns=columns)
    
    # Convert timestamp to datetime
    df['ts'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    
    # Select required columns (no taker volumes)
    df = df[['ts', 'open', 'high', 'low', 'close', 'base_vol', 'quote_vol']]
    
    # Convert string fields to float
    float_cols = ['open', 'high', 'low', 'close', 'base_vol', 'quote_vol']
    df[float_cols] = df[float_cols].astype(float)
    
    return df


async def fetch_and_parse_oi_history(
    session: aiohttp.ClientSession,
    symbol: str,
    period: str = "5m",
    limit: int = 500,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None
) -> pd.DataFrame:
    """
    Binance Open Interest History verisini çeker ve parse eder.
    
    Args:
        session: aiohttp session
        symbol: Trading pair (e.g., "BTCUSDT")
        period: Aggregation period (e.g., "5m")
        limit: Number of points, max 500
        start_time: Start time in milliseconds (Unix timestamp)
        end_time: End time in milliseconds (Unix timestamp)
    
    Returns:
        pd.DataFrame: Parsed DataFrame with required schema
    """
    # Binance Futures Open Interest History API endpoint - rate limiting uygulanır
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    params = {
        "symbol": symbol,
        "period": period,
        "limit": limit
    }
    
    if start_time is not None:
        params["startTime"] = start_time
    if end_time is not None:
        params["endTime"] = end_time
    
    async with session.get(url, params=params) as response:
        if response.status != 200:
            text = await response.text()
            raise Exception(f"API error: {response.status} - {text}")
        
        data = await response.json()
        if not data:
            return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    # Convert timestamp to datetime
    df['ts'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    
    # Map to required columns
    df['oi'] = df['sumOpenInterestValue'].astype(float)
    df = df[['ts', 'oi']]
    
    return df





# ============================================================================
# VERİ ÇEKME VE KAYDETME KATMANI
# ============================================================================

async def fetch_and_save_data_for_symbol(
    session: aiohttp.ClientSession,
    client: core_data.DataClient,
    symbol_row: pd.Series,
    symbol_id: int,
    start_ts: datetime,
    end_ts: datetime
) -> tuple[str, bool]:
    """
    Tek bir sembol için son 30 günlük veriyi çek ve veritabanına kaydet.
    
    Args:
        session: aiohttp session for API calls
        client: Database client for upsert operations
        symbol: Symbol name (e.g., "BTCUSDT")
        start_ts: Start timestamp
        end_ts: End timestamp
    
    Returns:
        tuple[str, bool]: (symbol, success)
    """
    try:
        # Rate limit savunması: her sembol arasında nefes alma payı
        await asyncio.sleep(0.1)
        
        # Symbol bilgilerini al
        symbol_name = symbol_row['symbol']
        # symbol_id artık parametre olarak geliyor, DataFrame index'inden hesaplamaya gerek yok
        
        # Perp ve spot sembol isimlerini al
        perp_symbol = symbol_row.get('perp_symbol', symbol_name)
        spot_symbol = symbol_row.get('spot_symbol', symbol_name)
        
        # Timestamp'leri milisaniyeye çevir
        start_ms = int(start_ts.timestamp() * 1000)
        end_ms = int(end_ts.timestamp() * 1000)
        
        print(f"Veri çekiliyor: {symbol_name} ({start_ts.strftime('%Y-%m-%d')} - {end_ts.strftime('%Y-%m-%d')})")
        
        # ========================================================================
        # FUTURES KLINES - Şartlı çekme
        # ========================================================================
        perp_df = pd.DataFrame()  # Varsayılan olarak boş
        
        if pd.notna(perp_symbol) and perp_symbol:
            print(f"  Futures klines çekiliyor: {perp_symbol}")
            perp_dfs = []
            current_start = start_ms
            
            while current_start < end_ms:
                current_end = min(current_start + (1500 * 5 * 60 * 1000), end_ms)  # 1500 * 5min in ms
                
                df = await fetch_and_parse_futures_klines(
                    session, perp_symbol, "5m", current_start, current_end, 1500
                )
                
                if not df.empty:
                    perp_dfs.append(df)
                
                # Rate limiting: Futures API için
                await asyncio.sleep(0.1)
                
                # Bir sonraki döngü için current_start'ı güncelle
                current_start = current_end + 1  # Bir sonraki milisaniyeden başla
            
            # Tüm futures DataFrame'leri birleştir
            perp_df = pd.concat(perp_dfs, ignore_index=True) if perp_dfs else pd.DataFrame()
            if not perp_df.empty:
                perp_df = perp_df.drop_duplicates(subset=['ts']).sort_values('ts')
                perp_df['symbol_id'] = symbol_id
        else:
            print(f"  Futures klines atlandı: perp_symbol tanımlanmamış")
        
        # ========================================================================
        # SPOT KLINES - Şartlı çekme
        # ========================================================================
        spot_df = pd.DataFrame()  # Varsayılan olarak boş
        
        if pd.notna(spot_symbol) and spot_symbol:
            print(f"  Spot klines çekiliyor: {spot_symbol}")
            spot_dfs = []
            current_start = start_ms
            
            while current_start < end_ms:
                current_end = min(current_start + (1000 * 5 * 60 * 1000), end_ms)  # 1000 * 5min in ms
                
                df = await fetch_and_parse_spot_klines(
                    session, spot_symbol, "5m", current_start, current_end, 1000
                )
                
                if not df.empty:
                    spot_dfs.append(df)
                
                # Rate limiting: Spot API için
                await asyncio.sleep(0.1)
                
                # Bir sonraki döngü için current_start'ı güncelle
                current_start = current_end + 1  # Bir sonraki milisaniyeden başla
            
            # Tüm spot DataFrame'leri birleştir
            spot_df = pd.concat(spot_dfs, ignore_index=True) if spot_dfs else pd.DataFrame()
            if not spot_df.empty:
                spot_df = spot_df.drop_duplicates(subset=['ts']).sort_values('ts')
                spot_df['symbol_id'] = symbol_id
        else:
            print(f"  Spot klines atlandı: spot_symbol tanımlanmamış")
        
        # ========================================================================
        # OPEN INTEREST - Şartlı çekme (sadece futures semboller için)
        # ========================================================================
        oi_df = pd.DataFrame()  # Varsayılan olarak boş
        
        if pd.notna(perp_symbol) and perp_symbol:
            print(f"  Open Interest çekiliyor: {perp_symbol}")
            oi_dfs = []
            current_start = start_ms
            
            while current_start < end_ms:
                current_end = min(current_start + (500 * 5 * 60 * 1000), end_ms)  # 500 * 5min in ms
                
                df = await fetch_and_parse_oi_history(
                    session, perp_symbol, "5m", 500, current_start, current_end
                )
                
                if not df.empty:
                    oi_dfs.append(df)
                
                # Rate limiting: OI API için daha konservatif
                await asyncio.sleep(0.2)
                
                # Bir sonraki döngü için current_start'ı güncelle
                current_start = current_end + 1  # Bir sonraki milisaniyeden başla
            
            # Tüm OI DataFrame'leri birleştir
            oi_df = pd.concat(oi_dfs, ignore_index=True) if oi_dfs else pd.DataFrame()
            if not oi_df.empty:
                oi_df = oi_df.drop_duplicates(subset=['ts']).sort_values('ts')
                oi_df['symbol_id'] = symbol_id
        else:
            print(f"  Open Interest atlandı: perp_symbol tanımlanmamış")
        
        # Veri kontrolü
        if perp_df.empty and spot_df.empty and oi_df.empty:
            print(f"BİLGİ: {symbol_name} için hiç veri bulunamadı")
            return symbol_name, True
        
        # --- Tip Zorlama ---
        if not perp_df.empty:
            perp_df['symbol_id'] = perp_df['symbol_id'].astype('int32')
            # Diğer tüm sayısal kolonların float olduğundan emin ol
            for col in ['open', 'high', 'low', 'close', 'base_vol', 'quote_vol', 'taker_buy_base_vol', 'taker_buy_quote_vol']:
                if col in perp_df.columns:
                    perp_df[col] = perp_df[col].astype('float64')

        if not spot_df.empty:
            spot_df['symbol_id'] = spot_df['symbol_id'].astype('int32')
            for col in ['open', 'high', 'low', 'close', 'base_vol', 'quote_vol']:
                if col in spot_df.columns:
                    spot_df[col] = spot_df[col].astype('float64')

        if not oi_df.empty:
            oi_df['symbol_id'] = oi_df['symbol_id'].astype('int32')
            if 'oi' in oi_df.columns:
                oi_df['oi'] = oi_df['oi'].astype('float64')
        
        # --- CVD HESAPLA VE KAYDET ---
        cvd_df = pd.DataFrame()
        if not perp_df.empty and 'taker_buy_base_vol' in perp_df.columns and 'base_vol' in perp_df.columns:
            print(f"  CVD deltası hesaplanıyor: {symbol_name}")
            cvd_df = perp_df[['symbol_id', 'ts']].copy()
            cvd_df['cvd_delta_base'] = 2 * perp_df['taker_buy_base_vol'] - perp_df['base_vol']
            cvd_df['cvd_delta_base'] = cvd_df['cvd_delta_base'].astype('float64')
        # --- CVD HESAPLA VE KAYDET SONU ---
        
        # --- Tip Zorlama Sonu ---

        # ========================================================================
        # VERİTABANINA YAZ
        # ========================================================================
        perp_rows, spot_rows, oi_rows, cvd_rows = 0, 0, 0, 0

        if not perp_df.empty:
            perp_rows = await client.upsert_dataframe(perp_df, 'klines_perp_5m', ['symbol_id', 'ts'])
        
        if not spot_df.empty:
            spot_rows = await client.upsert_dataframe(spot_df, 'klines_spot_5m', ['symbol_id', 'ts'])
        
        if not oi_df.empty:
            oi_rows = await client.upsert_dataframe(oi_df, 'oi_5m', ['symbol_id', 'ts'])
        
        if not cvd_df.empty:
            cvd_rows = await client.upsert_dataframe(cvd_df, 'cvd_5m', ['symbol_id', 'ts'])
        
        print(f"OK: {symbol_name} -> {perp_rows} perp, {spot_rows} spot, {oi_rows} OI, {cvd_rows} CVD bar yazıldı.")
        return symbol_name, True
        
    except Exception as e:
        print(f"HATA: {symbol_name} -> {e}")
        return symbol_name, False


async def main():
    """Ana fonksiyon - Son 30 günlük veriyi paralel olarak çek ve kaydet"""
    print("QuantZilla - Data Ingestor Başlatılıyor...")
    print("=" * 60)
    
    # DataClient'ı oluştur
    client = core_data.get_db_client()
    if not client:
        print("Veritabanı bağlantısı kurulamadı, çıkılıyor")
        sys.exit(1)
    
    # Tablo şemalarını yükle
    await client.initialize()
    
    # Zaman aralığını belirle (son 30 gün)
    end_ts = datetime.now(timezone.utc)
    start_ts = end_ts - timedelta(days=30)
    
    print(f"Zaman aralığı: {start_ts.strftime('%Y-%m-%d %H:%M')} - {end_ts.strftime('%Y-%m-%d %H:%M')}")
    
    # Sembolleri al (tam DataFrame ve harita ile)
    symbols_df, name_to_id = await core_data.load_enabled_symbols_async()
    if symbols_df.empty:
        print("Hiç sembol bulunamadı, çıkılıyor")
        sys.exit(1)
    
    print(f"\n{len(symbols_df)} sembol için batch modunda veri çekiliyor...")
    
    # Batch boyutunu belirle (rate limiting için konservatif)
    BATCH_SIZE = 5
    symbol_batches = [symbols_df.iloc[i:i + BATCH_SIZE] for i in range(0, len(symbols_df), BATCH_SIZE)]
    
    print(f"Toplam {len(symbol_batches)} batch oluşturuldu (her batch {BATCH_SIZE} sembol)")
    
    # Rate limit'e dayanıklı session oluştur
    retry_options = ExponentialRetry(
        attempts=3,
        start_timeout=1,
        max_timeout=30,
        factor=2,
        statuses=[429, 500, 502, 503, 504]
    )
    
    all_results = []
    
    async with RetryClient(
        retry_options=retry_options,
        timeout=aiohttp.ClientTimeout(total=60)
    ) as session:
        # Semaphore ile eşzamanlı görev sayısını sınırla (konservatif - rate limiting için)
        semaphore = asyncio.Semaphore(3)
        
        async def process_symbol_with_semaphore(symbol, symbol_id):
            async with semaphore:
                return await fetch_and_save_data_for_symbol(session, client, symbol, symbol_id, start_ts, end_ts)
        
        # Her batch için döngü
        for batch_idx, batch_symbols_df in enumerate(symbol_batches, 1):
            batch_symbol_names = batch_symbols_df['symbol'].tolist()
            print(f"\n--- BATCH {batch_idx}/{len(symbol_batches)} İŞLENİYOR ---")
            print(f"Semboller: {', '.join(batch_symbol_names)}")
            
            # Bu batch için async task'ları oluştur
            batch_tasks = []
            for _, symbol_row in batch_symbols_df.iterrows():
                symbol_name = symbol_row['symbol']
                symbol_id = name_to_id.get(symbol_name)
                if symbol_id is None:
                    print(f"UYARI: {symbol_name} için haritada ID bulunamadı, atlanıyor.")
                    continue
                # process_symbol_with_semaphore fonksiyonuna artık symbol_id'yi de gönderin
                batch_tasks.append(
                    process_symbol_with_semaphore(symbol_row, symbol_id)
                )
            
            # Bu batch'i paralel olarak çalıştır
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            all_results.extend(batch_results)
            
            print(f"Batch {batch_idx} tamamlandı: {len(batch_symbols_df)} sembol işlendi")
            
            # Son batch değilse soğuma periyodu bekle (rate limiting için)
            if batch_idx < len(symbol_batches):
                print(f"90 saniye soğuma periyodu bekleniyor...")
                await asyncio.sleep(90)
            else:
                print("Son batch tamamlandı - soğuma periyodu gerekmiyor")
    
    # Tüm sonuçları birleştir
    results = all_results
    
    # Sonuçları say
    success_count = 0
    error_count = 0
    
    for result in results:
        if isinstance(result, Exception):
            print(f"Exception: {result}")
            error_count += 1
        elif isinstance(result, tuple) and len(result) == 2:
            symbol, success = result
            if success:
                success_count += 1
            else:
                error_count += 1
        else:
            error_count += 1
    
    # Sonuç
    print("\n" + "=" * 60)
    print(f"İngest tamamlandı!")
    print(f"Başarılı: {success_count}")
    if error_count > 0:
        print(f"Hatalı: {error_count}")
    
    if error_count == 0:
        print("Tüm veriler başarıyla çekildi!")
        sys.exit(0)
    else:
        print("Bazı sembollerde hata oluştu")
        sys.exit(1)


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print(".env dosyası yüklendi")
    except ImportError:
        print("dotenv kütüphanesi bulunamadı")
    
    asyncio.run(main())
