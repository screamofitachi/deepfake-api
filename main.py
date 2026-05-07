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

from fastapi import FastAPI

from model_loader import load_model, get_model_info, is_model_loaded


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
    }