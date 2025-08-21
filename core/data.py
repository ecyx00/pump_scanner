import os
import asyncio
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Dict
from datetime import datetime
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import Table, MetaData


@dataclass
class DbConfig:
    dsn: str


class DataClient:
    """Async PostgreSQL data client for 5m OHLCV, OI and CVD access.

    Tables expected (per database_schema.md):
      - klines_perp_5m(symbol_id, ts, open, high, low, close, base_vol, quote_vol, taker_buy_base_vol, taker_buy_quote_vol)
      - klines_spot_5m(symbol_id, ts, open, high, low, close, base_vol, quote_vol)
      - oi_5m(symbol_id, ts, oi)
      - cvd_5m(symbol_id, ts, cvd_delta_base)
    """

    def __init__(self, engine: AsyncEngine):
        self._engine = engine
        self._tables = {}
        self._metadata = MetaData()

    @classmethod
    def from_env(cls) -> "DataClient":
        dsn = os.getenv("POSTGRES_DSN")
        if not dsn:
            raise RuntimeError("POSTGRES_DSN env var is not set. Configure DB or provide offline data.")
        if dsn.startswith("postgresql://"):
            # Enforce async driver
            dsn = dsn.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif not dsn.startswith("postgresql+asyncpg://"):
            dsn = f"postgresql+asyncpg://{dsn}"
        engine = create_async_engine(dsn, pool_pre_ping=True)
        return cls(engine)

    async def initialize(self):
        """Veritabanı tablo şemalarını yükler."""
        async with self._engine.connect() as conn:
            # Tablo şemalarını reflect ile yükle (senkron metodu asenkron olarak çalıştır)
            await conn.run_sync(self._metadata.reflect, only=[
                'klines_perp_5m', 'klines_spot_5m', 'oi_5m'
            ])
            
            # Yüklenen tabloları sözlükte sakla
            for table_name in ['klines_perp_5m', 'klines_spot_5m', 'oi_5m']:
                if table_name in self._metadata.tables:
                    self._tables[table_name] = self._metadata.tables[table_name]
                else:
                    raise RuntimeError(f"Tablo {table_name} bulunamadı")

    async def fetch_symbols(self) -> pd.DataFrame:
        query = text(
            """
            SELECT id, name, market, exchange, base_asset, quote_asset, active
            FROM symbols
            WHERE active = TRUE
            """
        )
        async with self._engine.connect() as conn:
            result = await conn.execute(query)
            df = pd.DataFrame(result.mappings().all())
        return df

    async def fetch_perp_klines_5m(self, symbol_ids: Sequence[int], start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
        query = text(
            """
            SELECT symbol_id, ts, open, high, low, close, base_vol, quote_vol, taker_buy_base_vol, taker_buy_quote_vol
            FROM klines_perp_5m
            WHERE symbol_id = ANY(:symbol_ids) AND ts >= :start_ts AND ts <= :end_ts
            ORDER BY symbol_id, ts
            """
        )
        async with self._engine.connect() as conn:
            result = await conn.execute(query, {"symbol_ids": list(symbol_ids), "start_ts": start_ts, "end_ts": end_ts})
            df = pd.DataFrame(result.mappings().all())
        return df

    async def fetch_spot_klines_5m(self, symbol_ids: Sequence[int], start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
        query = text(
            """
            SELECT symbol_id, ts, open, high, low, close, base_vol, quote_vol
            FROM klines_spot_5m
            WHERE symbol_id = ANY(:symbol_ids) AND ts >= :start_ts AND ts <= :end_ts
            ORDER BY symbol_id, ts
            """
        )
        async with self._engine.connect() as conn:
            result = await conn.execute(query, {"symbol_ids": list(symbol_ids), "start_ts": start_ts, "end_ts": end_ts})
            df = pd.DataFrame(result.mappings().all())
        return df

    async def fetch_oi_5m(self, symbol_ids: Sequence[int], start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
        query = text(
            """
            SELECT symbol_id, ts, oi
            FROM oi_5m
            WHERE symbol_id = ANY(:symbol_ids) AND ts >= :start_ts AND ts <= :end_ts
            ORDER BY symbol_id, ts
            """
        )
        async with self._engine.connect() as conn:
            result = await conn.execute(query, {"symbol_ids": list(symbol_ids), "start_ts": start_ts, "end_ts": end_ts})
            df = pd.DataFrame(result.mappings().all())
        return df

    async def fetch_cvd_5m(self, symbol_ids: Sequence[int], start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
        query = text(
            """
            SELECT symbol_id, ts, cvd_delta_base
            FROM cvd_5m
            WHERE symbol_id = ANY(:symbol_ids) AND ts >= :start_ts AND ts <= :end_ts
            ORDER BY symbol_id, ts
            """
        )
        async with self._engine.connect() as conn:
            result = await conn.execute(query, {"symbol_ids": list(symbol_ids), "start_ts": start_ts, "end_ts": end_ts})
            df = pd.DataFrame(result.mappings().all())
        return df

    async def fetch_average_daily_volume_bulk(self, symbol_ids: Sequence[int], days: int = 30) -> pd.DataFrame:
        """Return average daily quote volume for multiple symbols in a single query.

        Computes: sum(quote_vol) over last N days, divided by N for each symbol.
        """
        query = text(
            """
            SELECT 
                symbol_id, 
                COALESCE(SUM(quote_vol), 0.0) / :days AS adv
            FROM klines_perp_5m
            WHERE symbol_id = ANY(:symbol_ids) 
                AND ts >= NOW() - (:days || ' days')::interval
            GROUP BY symbol_id
            """
        )
        async with self._engine.connect() as conn:
            result = await conn.execute(query, {"symbol_ids": list(symbol_ids), "days": days})
            df = pd.DataFrame(result.mappings().all())
            return df

    async def upsert_dataframe(self, df: pd.DataFrame, table_name: str, pk_cols: list[str]):
        """
        Generic upsert for a pandas DataFrame using asyncpg's fast copy_to_table.
        """
        if df.empty:
            return 0

        # SQL injection koruması: tablo adı ve kolon whitelist kontrolü
        allowed_tables = {'klines_perp_5m', 'klines_spot_5m', 'oi_5m', 'cvd_5m'}
        if table_name not in allowed_tables:
            raise ValueError(f"Geçersiz tablo adı: {table_name}")
        
        # Kolon whitelist kontrolü
        allowed_columns = {
            'symbol_id', 'ts', 'open', 'high', 'low', 'close', 
            'base_vol', 'quote_vol', 'taker_buy_base_vol', 'taker_buy_quote_vol', 'oi', 'cvd_delta_base'
        }
        
        # Primary key kolonlarını kontrol et
        if not all(col in allowed_cols for col in pk_cols):
            invalid_pk_cols = [col for col in pk_cols if col not in allowed_cols]
            raise ValueError(f"Geçersiz primary key kolonları: {invalid_pk_cols}")
        
        # DataFrame kolon isimlerinin tablo ile eşleştiğinden emin ol
        # Bu, bir sonraki adımda gereklidir
        temp_table_name = f"temp_{table_name}"
        
        # DataFrame kolonlarını whitelist ile kontrol et
        df_columns = set(df.columns)
        if not df_columns.issubset(allowed_cols):
            invalid_cols = df_columns - allowed_cols
            raise ValueError(f"DataFrame'de geçersiz kolonlar: {invalid_cols}")
        
        # Pandas'taki NaN'ları None'a (NULL) çevir
        df = df.replace([np.inf, -np.inf], np.nan).where(pd.notnull(df), None)

        # Sütun sırasını veritabanı şemasıyla eşleştirmek için tablo şemasını al (parametrized query)
        table_cols_query = text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = :table_name ORDER BY ordinal_position;
        """)

        async with self._engine.connect() as conn:
            # Gerçek tablo kolon sırasını al (parametrized query)
            result = await conn.execute(table_cols_query, {"table_name": table_name})
            table_columns = [row[0] for row in result.fetchall()]
            
            # Tablo kolonlarını whitelist ile kontrol et
            if not all(col in allowed_cols for col in table_columns):
                invalid_cols = [col for col in table_columns if col not in allowed_cols]
                raise ValueError(f"Veritabanı tablosunda geçersiz kolonlar: {invalid_cols}")
            
            # DataFrame'i doğru sıraya getir
            df_ordered = df[table_columns]
            
            # Veriyi kayıtlara çevir
            records = list(df_ordered.itertuples(index=False, name=None))

            # asyncpg'nin copy_records_to_table'ını kullan
            # Bu, SQLAlchemy'den çok daha hızlı ve daha sağlamdır.
            # Güvenlik: temp table adı ve tablo adı whitelist'te
            create_temp_query = text("CREATE TEMP TABLE :temp_table (LIKE :table)")
            await conn.run_sync(
                lambda sync_conn: sync_conn.execute(
                    create_temp_query, {"temp_table": temp_table_name, "table": table_name}
                )
            )
            
            raw_conn = await conn.get_raw_connection()
            # Güvenlik: temp table adı whitelist'te, columns da whitelist'te
            await raw_conn.driver_connection.copy_records_to_table(
                temp_table_name, records=records, columns=table_columns
            )

            # Şimdi, temp tablodan ana tabloya upsert yap
            # Güvenlik: Tüm değerler parametrized query ile
            update_cols_str = ", ".join([f"{col} = EXCLUDED.{col}" for col in table_columns if col not in pk_cols])
            pk_cols_str = ", ".join(pk_cols)

            upsert_query = text("""
                INSERT INTO :table_name (""" + ", ".join([f":col_{i}" for i in range(len(table_columns))]) + """)
                SELECT * FROM :temp_table
                ON CONFLICT (""" + pk_cols_str + """) DO UPDATE SET """ + update_cols_str + """;
            """)
            
            # Parametreleri hazırla
            params = {"table_name": table_name, "temp_table": temp_table_name}
            for i, col in enumerate(table_columns):
                params[f"col_{i}"] = col
            
            result = await conn.execute(upsert_query, params)
            # Güvenlik: temp table silme (parametrized query)
            drop_temp_query = text("DROP TABLE :temp_table")
            await conn.run_sync(lambda sync_conn: sync_conn.execute(drop_temp_query, {"temp_table": temp_table_name}))
            
            return result.rowcount


async def load_enabled_symbols_async(symbols_csv_path: str = "symbols.csv") -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Enabled sembolleri yükler ve veritabanındaki gerçek ID'lerle mapping oluşturur.
    
    Offline modda CSV dosyasındaki sıralamayı kullanır.
    Online modda veritabanındaki gerçek ID'leri kullanır.
    """
    df = pd.read_csv(symbols_csv_path)
    df = df[df["enabled"] == 1]
    
    if df.empty:
        return df, {}
    
    # Veritabanı bağlantısını kontrol et
    client = get_db_client()
    if not client:
        print("Veritabanı bağlantısı yok, offline modda çalışılıyor...")
        # Offline mod: CSV sıralamasını kullan
        mapping = {row["symbol"]: idx + 1 for idx, row in df.iterrows()}
        print(f"Offline modda {len(mapping)} sembol ID'si oluşturuldu")
        return df, mapping
    
    try:
        # Online mod: Veritabanından gerçek ID'leri çek
        symbols_from_db = await client.fetch_symbols()
        
        # CSV'deki sembolleri veritabanındaki ID'lerle eşleştir
        mapping = {}
        for _, csv_row in df.iterrows():
            symbol_name = csv_row["symbol"]
            # Veritabanında bu sembolü ara
            db_match = symbols_from_db[symbols_from_db["name"] == symbol_name]
            if not db_match.empty:
                mapping[symbol_name] = int(db_match.iloc[0]["id"])
            else:
                print(f"UYARI: {symbol_name} veritabanında bulunamadı, atlanıyor")
        
        print(f"Veritabanından {len(mapping)} sembol ID'si alındı")
        return df, mapping
        
    except Exception as e:
        print(f"Veritabanı hatası: {e}, offline moda geçiliyor...")
        # Hata durumunda offline moda geç
        mapping = {row["symbol"]: idx + 1 for idx, row in df.iterrows()}
        print(f"Offline modda {len(mapping)} sembol ID'si oluşturuldu")
        return df, mapping


def get_db_client() -> Optional[DataClient]:
    try:
        return DataClient.from_env()
    except Exception:
        return None


# Senkron sarmalayıcılar kaldırıldı - çağıran betikler kendi asyncio.run() ile yönetmeli
# Bu fonksiyonlar artık mevcut değil:
# - get_perp_klines
# - get_spot_klines  
# - get_oi_series
# - get_cvd_series
# - _run
