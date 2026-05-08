"""
Predictor - Deepfake Detection
==============================
Bu modül, yüklenen görseli alıp model üzerinden tahmin yapar.

Pipeline:
1. Bytes → PIL Image
2. RGB'ye çevir (RGBA, grayscale gelirse normalize)
3. 224x224'e resize et (LANCZOS, en kaliteli)
4. NumPy array'e çevir, [0-255] HAM bırak (normalize ETME!)
5. Batch dimension ekle → (1, 224, 224, 3)
6. model.predict() → sigmoid çıktısı (0-1 arası)
7. Threshold uygula → label + confidence
"""

import time
import logging
from typing import Dict, Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from model_loader import (
    get_model,
    CLASS_NAMES,
    THRESHOLD,
)
from image_processor import preprocess_image


logger = logging.getLogger(__name__)


def predict(image_bytes: bytes) -> Dict[str, Any]:
    """
    Görsel bytes'ından deepfake tahmini yapar.
    
    Args:
        image_bytes: Görsel dosyasının byte verisi
    
    Returns:
        {
            "label": "Real" | "Deepfake",
            "confidence": float (0-1),
            "is_deepfake": bool,
            "raw_score": float (sigmoid output, 0-1),
            "processing_time_ms": float
        }
    
    Raises:
        ValueError: Görsel hatası
        RuntimeError: Model hatası
    """
    start_time = time.time()
    
    # 1. Preprocessing
    try:
        x = preprocess_image(image_bytes)
    except ValueError:
        raise  # Olduğu gibi yukarı fırlat
    except Exception as e:
        raise ValueError(f"Preprocessing hatası: {str(e)}")
    
    # 2. Modeli al (zaten yüklenmişse cache'ten döner)
    try:
        model = get_model()
    except Exception as e:
        raise RuntimeError(f"Model alınamadı: {str(e)}")
    
    # 3. Tahmin yap
    try:
        # verbose=0 → konsola log basmasın (API'de gereksiz)
        raw_output = model.predict(x, verbose=0)
        # Çıktı shape: (1, 1) → tek değer al: (1, 1) -> [0][0]
        raw_score = float(raw_output[0][0])
    except Exception as e:
        raise RuntimeError(f"Tahmin sırasında hata: {str(e)}")
    
    # 4. Sonucu yorumla
    is_deepfake = raw_score > THRESHOLD
    label_idx = 1 if is_deepfake else 0
    label = CLASS_NAMES[label_idx]
    
    # Confidence: 1'e ne kadar yakınsa Deepfake'e o kadar emin,
    # 0'a ne kadar yakınsa Real'e o kadar emin
    confidence = raw_score if is_deepfake else (1 - raw_score)
    
    # Süre hesapla
    processing_time_ms = (time.time() - start_time) * 1000
    
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "is_deepfake": is_deepfake,
        "raw_score": round(raw_score, 6),
        "processing_time_ms": round(processing_time_ms, 2),
    }


# Doğrudan çalıştırılırsa basit test
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    print("\n" + "=" * 60)
    print("Predictor Test")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("\nKullanım: python predictor.py <görsel_dosya_yolu>")
        print("\nÖrnek: python predictor.py test_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print(f"\n📂 Görsel okunuyor: {image_path}")
    
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        print(f"✓ Dosya boyutu: {len(image_bytes) / 1024:.1f} KB")
        print(f"\n🧠 Tahmin yapılıyor...")
        
        result = predict(image_bytes)
        
        print(f"\n{'=' * 60}")
        print(f"  SONUÇ")
        print(f"{'=' * 60}")
        print(f"  Sınıf       : {result['label']}")
        print(f"  Güven       : %{result['confidence'] * 100:.2f}")
        print(f"  Deepfake mi : {'EVET ⚠️' if result['is_deepfake'] else 'HAYIR ✓'}")
        print(f"  Ham skor    : {result['raw_score']:.6f}")
        print(f"  Süre        : {result['processing_time_ms']:.1f} ms")
        print(f"{'=' * 60}\n")
    
    except FileNotFoundError:
        print(f"\n❌ Dosya bulunamadı: {image_path}")
    except ValueError as e:
        print(f"\n❌ Görsel hatası: {e}")
    except RuntimeError as e:
        print(f"\n❌ Model hatası: {e}")
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")