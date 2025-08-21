# 🔒 Güvenlik Politikası

## Güvenlik Açığı Raporlama

Eğer bir güvenlik açığı keşfederseniz, lütfen **özel olarak** rapor edin:

- **Email**: security@quantzilla.com
- **GitHub Security**: [Security Advisories](https://github.com/yourusername/pump_scanner/security/advisories)

**Lütfen güvenlik açıklarını public issue olarak raporlamayın.**

## Güvenlik Önlemleri

### 1. Environment Variables
- Tüm API anahtarları ve veritabanı bağlantı bilgileri `.env` dosyasında saklanır
- `.env` dosyası `.gitignore` ile korunur ve GitHub'a yüklenmez
- Örnek konfigürasyon `env.example` dosyasında bulunur

### 2. Veritabanı Güvenliği
- SQL injection koruması için parametrized queries kullanılır
- Veritabanı bağlantıları async/await pattern ile yönetilir
- Connection pooling ve timeout'lar uygulanır

### 3. API Güvenliği
- Binance API rate limiting uygulanır
- Retry mekanizması exponential backoff ile
- Timeout'lar ve error handling

### 4. Kod Güvenliği
- Input validation tüm kullanıcı girdilerinde
- Error handling crash-safe
- Logging'de hassas bilgi yok

## Güvenlik Kontrol Listesi

- [ ] `.env` dosyası `.gitignore`'da
- [ ] API anahtarları kodda hard-code edilmemiş
- [ ] Veritabanı şifreleri açıkta değil
- [ ] SQL injection koruması aktif
- [ ] Rate limiting uygulanmış
- [ ] Error handling güvenli
- [ ] Logging'de hassas bilgi yok

## Güvenlik Güncellemeleri

Güvenlik güncellemeleri için:
1. Security advisory oluşturun
2. CVE numarası alın (gerekirse)
3. Patch'i test edin
4. Release notes'da güvenlik bilgisi ekleyin

---

**Güvenlik, kod kalitesi kadar önemlidir.** 🔒
