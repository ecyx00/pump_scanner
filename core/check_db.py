# core/check_db.py
import sys
import asyncio
from core.data import DataClient

async def main_async():
    """Async olarak veritabanı bağlantısını test eder."""
    print(">> Veritabanı bağlantısı test ediliyor...")
    try:
        client = DataClient.from_env()
        # Bağlantıyı test etmek için basit ve hızlı bir sorgu
        _ = await client.fetch_symbols()
        print("\n[SUCCESS] Veritabanı bağlantısı başarılı.")
        print(">> Konfigürasyon doğru. Sistem hazır.")
        return 0
    except Exception as e:
        print(f"\n[FAILURE] Veritabanı bağlantısı BAŞARISIZ OLDU.")
        print(f">> Hata: {e}")
        print(">> .env dosyasındaki POSTGRES_DSN değişkenini kontrol et.")
        print(">> Örnek: cp env.example .env && düzenle")
        return 1

def main():
    """Senkron wrapper - async main fonksiyonunu çalıştırır"""
    # Windows'ta asyncio için özel event loop politikası gerekebilir
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    return_code = asyncio.run(main_async())
    sys.exit(return_code)

if __name__ == "__main__":
    main()