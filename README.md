# Deepfake Detection API

Yüklenen yüz görsellerinin gerçek mi yoksa deepfake (yapay zeka ile üretilmiş sahte yüz) mi olduğunu tespit eden REST API.

## Proje Hakkında

Bu API, derin öğrenme modelleri kullanarak yüz görsellerini "Gerçek" veya "Sahte (Deepfake)" olarak sınıflandırır. Kullanıcının yüklediği görseli işler ve güven skoruyla birlikte sonucu JSON formatında döndürür.

## Teknolojiler

- **Python 3.13**
- **FastAPI** — REST API framework
- **Uvicorn** — ASGI sunucu
- **PyTorch / TensorFlow** — Derin öğrenme modeli (entegre edilecek)

## Kurulum

### 1. Repoyu Klonla

```bash
git clone <repo-url>
cd deepfake-api
```

### 2. Sanal Ortam Oluştur

```bash
python -m venv venv
```

### 3. Sanal Ortamı Aktif Et

**Windows (PowerShell):**
```bash
.\venv\Scripts\Activate.ps1
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### 4. Bağımlılıkları Kur

```bash
pip install -r requirements.txt
```

### 5. Sunucuyu Çalıştır

```bash
uvicorn main:app --reload
```

API artık `http://localhost:8000` adresinde çalışıyor.

## Endpoint'ler

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| GET | `/` | Ana sayfa, API durumu |
| GET | `/health` | Sağlık kontrolü |
| GET | `/docs` | Swagger UI dokümantasyonu |

## Geliştirme Aşaması

Bu proje aktif geliştirme aşamasındadır. Sprint planı:

- [x] **SCRUM-22**: Web Framework Kurulumu
- [ ] **SCRUM-23**: Model Entegrasyonu
- [ ] **SCRUM-24**: API Endpoint Geliştirme (`/predict`)
- [ ] **SCRUM-25**: Görüntü İşleme Fonksiyonu
- [ ] **SCRUM-26**: Hata Yönetimi
- [ ] **SCRUM-27**: CORS Ayarları

## Geliştirici

**Furkan İşık** — Backend & API Developer

## Lisans

Bu proje akademik amaçlı geliştirilmiştir.