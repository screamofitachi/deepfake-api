"""
Model Loader - Deepfake Detection API
=====================================
Bu modül, EfficientNetV2M deepfake tespit modelini bir kez yükler ve
uygulama yaşam döngüsü boyunca bellekte tutar.

Model: best_efficientnetv2m.keras
Test Accuracy: 98.07%
Inference: ~20 ms/görsel (T4 GPU), ~100-200 ms (CPU)
"""

import os
import logging
from pathlib import Path
from typing import Optional

import tensorflow as tf

# Logging yapılandırması
logger = logging.getLogger(__name__)

# Model dosyasının yolu - proje köküne göre relative
MODEL_PATH = Path(__file__).parent / "models" / "best_efficientnetv2m.keras"

# Sınıf etiketleri ve eşik değeri (Muhammet'in model_card.md dosyasından)
CLASS_NAMES = ["Real", "Deepfake"]
THRESHOLD = 0.5
INPUT_SIZE = (224, 224)

# Global model nesnesi - bir kere yüklenecek
_model: Optional[tf.keras.Model] = None


def load_model() -> tf.keras.Model:
    """
    Modeli diskten yükler ve global olarak saklar.
    
    Returns:
        Yüklenmiş Keras modeli
    
    Raises:
        FileNotFoundError: Model dosyası bulunamazsa
        RuntimeError: Model yüklenirken hata olursa
    """
    global _model
    
    if _model is not None:
        logger.info("Model zaten bellekte, yeniden yüklenmiyor.")
        return _model
    
    if not MODEL_PATH.exists():
        error_msg = f"Model dosyası bulunamadı: {MODEL_PATH}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info(f"Model yükleniyor: {MODEL_PATH}")
    logger.info(f"Dosya boyutu: {MODEL_PATH.stat().st_size / (1024 * 1024):.1f} MB")
    
    try:
        _model = tf.keras.models.load_model(str(MODEL_PATH))
        logger.info("Model başarıyla yüklendi.")
        logger.info(f"Toplam parametre: {_model.count_params():,}")
        return _model
    except Exception as e:
        error_msg = f"Model yüklenemedi: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def get_model() -> tf.keras.Model:
    """
    Yüklenmiş modeli döndürür. Henüz yüklenmemişse load_model() çağırır.
    
    Returns:
        Yüklenmiş Keras modeli
    """
    if _model is None:
        return load_model()
    return _model


def is_model_loaded() -> bool:
    """
    Model yüklü mü kontrolü.
    
    Returns:
        True: model bellekte, False: değil
    """
    return _model is not None


def get_model_info() -> dict:
    """
    Model hakkında bilgi döndürür (health check ve API dokümantasyonu için).
    
    Returns:
        Model meta bilgileri
    """
    return {
        "model_name": "EfficientNetV2M Deepfake Detector",
        "model_path": str(MODEL_PATH.name),
        "is_loaded": is_model_loaded(),
        "input_size": list(INPUT_SIZE),
        "classes": CLASS_NAMES,
        "threshold": THRESHOLD,
        "framework": "TensorFlow/Keras",
        "test_accuracy": 0.9807,
    }


# Doğrudan çalıştırılırsa basit bir test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    print("\n" + "=" * 60)
    print("Model Loader Test")
    print("=" * 60)
    
    try:
        model = load_model()
        print(f"\n✅ Model yüklendi!")
        print(f"   Input shape : {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Parametre   : {model.count_params():,}")
        print(f"\nModel bilgisi:")
        for key, value in get_model_info().items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"\n❌ Hata: {e}")