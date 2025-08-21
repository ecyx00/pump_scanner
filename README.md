# 🚀 QuantZilla - Pump Scanner

QuantZilla, kripto para piyasalarında volume ve open interest spike'larını tespit eden gelişmiş bir trading strateji sistemidir. Python tabanlı, yüksek performanslı ve veri odaklı bir yaklaşım kullanır.

## ✨ Özellikler

- **Volume & OI Spike Detection**: 5 dakikalık timeframe'de anormal aktivite tespiti
- **Multi-Asset Support**: Binance USDⓈ-M Futures ve Spot piyasaları
- **Real-time Data**: PostgreSQL veritabanı ile hızlı veri erişimi
- **Backtesting**: Strateji performans analizi
- **Configurable Parameters**: YAML tabanlı konfigürasyon sistemi

## 🏗️ Mimari

```
pump_scanner/
├── core/                    # Ana sistem modülleri
│   ├── data.py            # Veritabanı client'ı
│   ├── ingestor.py        # Veri çekme ve kaydetme
│   ├── backtest.py        # Backtesting motoru
│   └── check_db.py        # Veritabanı bağlantı testi
├── strategies/             # Trading stratejileri
│   └── v1_volume_oi_spike/ # Volume/OI spike stratejisi
├── md/                     # Dokümantasyon
├── symbols.csv             # Sembol listesi
└── run_all.py             # Ana çalıştırma scripti
```

## 🚀 Kurulum

### 1. Gereksinimler

- Python 3.10+
- PostgreSQL 12+
- TimescaleDB (opsiyonel, performans için)

### 2. Repository'yi Klonlayın

```bash
git clone https://github.com/yourusername/pump_scanner.git
cd pump_scanner
```

### 3. Virtual Environment Oluşturun

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# veya
.venv\Scripts\activate     # Windows
```

### 4. Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

### 5. Environment Variables Ayarlayın

```bash
cp env.example .env
# .env dosyasını düzenleyin ve veritabanı bilgilerinizi girin
```

## 📊 Veritabanı Kurulumu

### PostgreSQL + TimescaleDB

```sql
-- Veritabanı oluştur
CREATE DATABASE pump_scanner;

-- TimescaleDB extension'ı etkinleştir
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Tabloları oluştur
CREATE TABLE symbols (
    id SERIAL PRIMARY KEY,
    name VARCHAR(20) UNIQUE NOT NULL,
    market VARCHAR(10) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    base_asset VARCHAR(10) NOT NULL,
    quote_asset VARCHAR(10) NOT NULL,
    active BOOLEAN DEFAULT TRUE
);

CREATE TABLE klines_perp_5m (
    symbol_id INTEGER REFERENCES symbols(id),
    ts TIMESTAMPTZ NOT NULL,
    open DECIMAL(20,8) NOT NULL,
    high DECIMAL(20,8) NOT NULL,
    low DECIMAL(20,8) NOT NULL,
    close DECIMAL(20,8) NOT NULL,
    base_vol DECIMAL(30,8) NOT NULL,
    quote_vol DECIMAL(30,8) NOT NULL,
    taker_buy_base_vol DECIMAL(30,8) NOT NULL,
    taker_buy_quote_vol DECIMAL(30,8) NOT NULL,
    PRIMARY KEY (symbol_id, ts)
);

-- TimescaleDB hypertable oluştur
SELECT create_hypertable('klines_perp_5m', 'ts', chunk_time_interval => INTERVAL '1 day');

-- Index'ler ekle
CREATE INDEX idx_klines_perp_5m_symbol_ts ON klines_perp_5m (symbol_id, ts DESC);
CREATE INDEX idx_klines_perp_5m_ts ON klines_perp_5m (ts DESC);
```

## 🎯 Kullanım

### 1. Veritabanı Bağlantısını Test Edin

```bash
python -m core.check_db
```

### 2. Veri Çekin (Backfill)

```bash
python -m core.ingestor
```

### 3. Stratejiyi Çalıştırın

```bash
python -m strategies.v1_volume_oi_spike.calculate
python -m strategies.v1_volume_oi_spike.analyze
```

### 4. Tüm Adımları Tek Seferde Çalıştırın

```bash
python run_all.py
```

## ⚙️ Konfigürasyon

Strateji parametreleri `strategies/v1_volume_oi_spike/config.yml` dosyasında tanımlanır:

```yaml
nz: 720           # Z-score window (5m bars)
np: 21            # Price thrust lookback (5m bars)
w_vol: 0.45       # Volume weight
w_oi: 0.35        # Open Interest weight
w_price: 0.20     # Price weight
trigger_percentile: 98  # Trigger threshold
cooldown_bars: 24      # Cooldown period
symbols_csv: "symbols.csv"
```

## 🔒 Güvenlik

- **Environment Variables**: Tüm hassas bilgiler `.env` dosyasında saklanır
- **Database Security**: Parametrized queries kullanılır
- **API Rate Limiting**: Binance API rate limit'leri uygulanır
- **Input Validation**: Tüm kullanıcı girdileri doğrulanır

## 📈 Strateji Mantığı

Volume & OI Spike stratejisi şu adımları takip eder:

1. **Data Collection**: 5 dakikalık OHLCV, OI ve CVD verileri toplanır
2. **Normalization**: Z-score tabanlı normalizasyon uygulanır
3. **Scoring**: Volume, OI ve price momentum ağırlıklı skorlama
4. **Signal Generation**: Percentile tabanlı trigger sistemi
5. **Cooldown**: False positive'leri önlemek için soğuma periyodu

## 🧪 Test

```bash
# Veritabanı bağlantısını test et
python -m core.check_db

# Strateji hesaplamalarını test et
python -m strategies.v1_volume_oi_spike.calculate

# Analiz sonuçlarını kontrol et
python -m strategies.v1_volume_oi_spike.analyze
```

## 📝 Logs

Sistem logları console'da görüntülenir. Hata durumlarında detaylı bilgi verilir.

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## ⚠️ Uyarı

Bu yazılım sadece eğitim ve araştırma amaçlıdır. Gerçek trading'de kullanmadan önce kapsamlı test yapın. Yazar, bu yazılımın kullanımından doğabilecek finansal kayıplardan sorumlu değildir.

## 📞 İletişim

- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Email**: your.email@example.com

---

**QuantZilla** - Alpha'nın Kod ile Buluştuğu Yer 🚀
