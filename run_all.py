#!/usr/bin/env python3
"""
QuantZilla - Tek Komutla Her Şey
Kullanım: python run_all.py
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def print_status(message, status="INFO"):
    colors = {
        "INFO": "\033[94m",    # Mavi
        "SUCCESS": "\033[92m", # Yeşil
        "WARNING": "\033[93m", # Sarı
        "ERROR": "\033[91m",   # Kırmızı
        "RESET": "\033[0m"     # Reset
    }
    print(f"{colors.get(status, '')}[{status}]{colors['RESET']} {message}")

def run_command(command, description):
    """Komutu çalıştır ve sonucu kontrol et"""
    print_status(f"{description}...", "INFO")
    try:
        # Environment variables'ları subprocess'e aktar
        env = os.environ.copy()
        result = subprocess.run(command, shell=True, check=True, capture_output=True, encoding='utf-8', errors='ignore', env=env)
        print_status(f"{description} başarılı!", "SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"{description} başarısız!", "ERROR")
        # stdout ve stderr'i birleştir çünkü hata bazen stdout'ta olabilir
        error_output = ""
        if e.stdout:
            error_output += f"STDOUT:\n{e.stdout}\n"
        if e.stderr:
            error_output += f"STDERR:\n{e.stderr}\n"
        if not error_output:
            error_output = "Hata detayı yok"
        
        print_status(f"Hata: {error_output}", "ERROR")
        print_status(f"Return Code: {e.returncode}", "ERROR")
        return False

def load_env():
    """Environment variables yükle"""
    print_status(".env dosyası yükleniyor...", "INFO")
    
    if not Path(".env").exists():
        print_status(".env dosyası bulunamadı, offline modda çalışılıyor...", "WARNING")
        return True  # Offline mod için True döndür
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        if not os.getenv("POSTGRES_DSN"):
            print_status("POSTGRES_DSN environment variable bulunamadı, offline modda çalışılıyor...", "WARNING")
            return True  # Offline mod için True döndür
            
        print_status(".env dosyası yüklendi", "SUCCESS")
        return True
    except ImportError:
        print_status("python-dotenv kurulu değil, offline modda çalışılıyor...", "WARNING")
        return True  # Offline mod için True döndür

def main():
    """Ana fonksiyon"""
    print_status("QuantZilla Başlatılıyor...", "INFO")
    print_status("=" * 50, "INFO")
    
    # 1. .env yükle - Tüm alt süreçler için merkezi yükleme
    if not load_env():
        sys.exit(1)
    
    # 2. Veritabanı bağlantısını test et (opsiyonel)
    if os.getenv("POSTGRES_DSN"):
        if not run_command("python -m core.check_db", "Veritabanı bağlantısı test ediliyor"):
            print_status("Veritabanı testi başarısız, offline modda devam ediliyor...", "WARNING")
    else:
        print_status("Veritabanı bağlantısı yok, offline modda çalışılıyor...", "WARNING")
    
    # 3. Son 30 günlük veri çek (Backfill) - sadece veritabanı varsa
    if os.getenv("POSTGRES_DSN"):
        if not run_command("python -m core.ingestor", "Son 30 günlük veri çekiliyor (Backfill)"):
            print_status("Veri çekme başarısız, offline modda devam ediliyor...", "WARNING")
    else:
        print_status("Veri çekme atlanıyor (offline mod)", "WARNING")
    
    # 4. Stratejiyi çalıştır
    print_status("Strateji çalıştırılıyor...", "INFO")
    
    if not run_command("python -m strategies.v1_volume_oi_spike.calculate", "Hesaplama adımı"):
        sys.exit(1)
    
    if not run_command("python -m strategies.v1_volume_oi_spike.analyze", "Analiz adımı"):
        sys.exit(1)
    
    # 5. Sonuç
    today = datetime.now().strftime("%Y-%m-%d")
    report_path = f"out/v1_volume_oi_spike/{today}/report.html"
    
    print_status("=" * 50, "SUCCESS")
    print_status("TÜM İŞLEMLER TAMAMLANDI!", "SUCCESS")
    print_status(f"Rapor: {report_path}", "SUCCESS")
    print_status("Raporu açmak için: start " + report_path, "INFO")

if __name__ == "__main__":
    main()
