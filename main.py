"""
Deepfake Detection API - Main Application
==========================================
FastAPI uygulaması. Modeli startup'ta yükler, endpoint'leri tanımlar.

Endpoint'ler:
- GET  /         → API durumu
- GET  /health   → Sağlık kontrolü (model yüklü mü)
- GET  /docs     → Swagger UI dokümantasyonu (otomatik)
- POST /predict  → Tahmin endpoint'i (SCRUM-24'te eklenecek)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from model_loader import load_model, get_model_info, is_model_loaded
from predictor import predict as run_prediction
from validators import validate_upload, get_validation_info


# ============================================================
# Logging Yapılandırması
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# Lifespan Event Handler
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Uygulama yaşam döngüsü:
    - Startup: Modeli belleğe yükle
    - Shutdown: Kaynakları temizle
    """
    # ====== STARTUP ======
    logger.info("=" * 60)
    logger.info("🚀 Deepfake Detection API başlatılıyor...")
    logger.info("=" * 60)
    
    try:
        logger.info("Model yükleniyor (bu işlem 10-30 saniye sürebilir)...")
        load_model()
        logger.info("✅ Model başarıyla yüklendi, API hazır!")
    except FileNotFoundError as e:
        logger.error(f"❌ Model dosyası bulunamadı: {e}")
        logger.error("API model olmadan başlatılıyor, /predict çalışmayacak.")
    except Exception as e:
        logger.error(f"❌ Model yüklenirken hata: {e}")
        logger.error("API model olmadan başlatılıyor, /predict çalışmayacak.")
    
    # API çalışırken burası beklemede kalır
    yield
    
    # ====== SHUTDOWN ======
    logger.info("=" * 60)
    logger.info("🛑 Deepfake Detection API kapatılıyor...")
    logger.info("=" * 60)


# ============================================================
# FastAPI Uygulaması
# ============================================================
app = FastAPI(
    title="Deepfake Detection API",
    description="Yüklenen yüz görsellerini gerçek/sahte (deepfake) olarak sınıflandıran REST API",
    version="0.2.0",
    lifespan=lifespan,
)

# ============================================================
# CORS Yapılandırması (SCRUM-27)
# ============================================================
# Frontend'in farklı domain/port'lardan API'ye erişebilmesi için
# CORS izinleri yapılandırıldı.

# İzin verilen origin'ler (frontend'in çalışacağı adresler)
ALLOWED_ORIGINS = [
    "http://localhost:3000",      # React varsayılan dev port
    "http://localhost:3001",      # React alternatif port
    "http://localhost:5173",      # Vite (Vue/React) varsayılan
    "http://localhost:5174",      # Vite alternatif
    "http://localhost:8080",      # Vue CLI varsayılan
    "http://localhost:4200",      # Angular varsayılan
    "http://127.0.0.1:3000",      # localhost'un IP versiyonu
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8080",
    # Production frontend domain'i eklendiğinde buraya yazılacak
    # Örnek: "https://deepfake-detector.example.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,           # İzinli adresler
    allow_credentials=True,                   # Cookie/auth header desteği
    allow_methods=["GET", "POST", "OPTIONS"], # İzinli HTTP metodları
    allow_headers=["*"],                      # Tüm header'lar (Content-Type vb.)
    max_age=600,                              # Preflight cache süresi (saniye)
)

logger.info(f"CORS yapılandırıldı: {len(ALLOWED_ORIGINS)} origin izinli")


# ============================================================
# Endpoint'ler
# ============================================================

@app.get("/")
def root():
    """
    API ana sayfa - genel durum bilgisi.
    """
    return {
        "message": "Deepfake Detection API çalışıyor!",
        "status": "active",
        "version": "0.2.0",
        "model_loaded": is_model_loaded(),
        "documentation": "/docs",
    }


@app.get("/health")
def health_check():
    """
    Sağlık kontrolü endpoint'i.
    
    Model yüklü mü ve API hazır mı bilgisini döndürür.
    Production ortamında load balancer'lar ve monitoring araçları
    bu endpoint'i düzenli olarak kontrol eder.
    """
    model_loaded = is_model_loaded()
    
    return {
    "status": "ok" if model_loaded else "degraded",
    "api_ready": True,
    "model_ready": model_loaded,
    "model_info": get_model_info() if model_loaded else None,
    "validation_rules": get_validation_info(),
    "cors_enabled": True,
    "allowed_origins": ALLOWED_ORIGINS,
}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Yüklenen görselin gerçek mi yoksa deepfake mi olduğunu tahmin eder.
    
    **Parametreler:**
    - `file`: Görsel dosyası (JPG, PNG, JPEG, max 10 MB)
    
    **Cevap (200 OK):**
```json
    {
        "label": "Real" | "Deepfake",
        "confidence": 0.99,
        "is_deepfake": false,
        "raw_score": 0.0029,
        "processing_time_ms": 387.4,
        "filename": "test.jpg"
    }
```
    
    **Hata Kodları:**
    - `400`: Dosya boş, bozuk veya geçersiz görsel
    - `413`: Dosya çok büyük (>10 MB)
    - `415`: Format desteklenmiyor (PDF, TXT vb.)
    - `503`: Model henüz yüklenmedi
    - `500`: Beklenmedik sunucu hatası
    """
    # 1. Model yüklü mü kontrol et
    if not is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model henüz yüklenmedi. Lütfen birkaç saniye sonra tekrar deneyin."
        )
    
    # 2. Dosyanın içeriğini oku
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Dosya okunamadı: {str(e)}"
        )
    
    # 3. Tüm validasyonları çalıştır (uzantı, MIME, boyut, içerik)
    # Hata varsa otomatik HTTPException fırlatır (415, 413, 400)
    validate_upload(
        filename=file.filename,
        content_type=file.content_type,
        image_bytes=image_bytes,
    )
    
    # 4. Tahmin yap
    try:
        result = run_prediction(image_bytes)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Görsel işlenemedi: {str(e)}"
        )
    except RuntimeError as e:
        logger.error(f"Model hatası: {e}")
        raise HTTPException(
            status_code=500,
            detail="Tahmin sırasında bir hata oluştu. Lütfen tekrar deneyin."
        )
    
    # 5. Dosya adını da response'a ekle
    result["filename"] = file.filename
    
    # 6. Başarılı cevabı dön
    return result