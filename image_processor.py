"""
Image Processor - Deepfake Detection
=====================================
Bu modül, gelen görselleri modelin eğitimde kullandığı formata dönüştüren
yardımcı fonksiyonları içerir.

Pipeline:
1. Bytes → PIL Image
2. Renk uzayını RGB'ye çevir (RGBA, grayscale gibi formatları normalize et)
3. Modelin eğitiminde kullanılan boyuta resize et (224x224)
4. NumPy array'e dönüştür [0-255] HAM aralığında bırak
5. Batch dimension ekle: (224, 224, 3) → (1, 224, 224, 3)

NOT: EfficientNetV2M built-in preprocessing içerir, ekstra normalization
(0-1 veya -1+1 aralığına çevirme) YAPMAYIN!
"""

import io
import logging
from typing import Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)


# ============================================================
# Sabitler
# ============================================================
TARGET_SIZE: Tuple[int, int] = (224, 224)
TARGET_CHANNELS: int = 3  # RGB
SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG"}


# ============================================================
# Yardımcı Fonksiyonlar
# ============================================================

def open_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Byte verisini PIL Image nesnesine dönüştürür.
    
    Args:
        image_bytes: Görsel dosyasının ham byte verisi
    
    Returns:
        PIL Image nesnesi
    
    Raises:
        ValueError: Görsel açılamazsa veya bozuksa
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # PIL lazy yüklediği için verify ile kontrol edelim
        img.verify()
        # verify() sonrası tekrar açmak gerekiyor
        img = Image.open(io.BytesIO(image_bytes))
        return img
    except UnidentifiedImageError:
        raise ValueError("Yüklenen dosya geçerli bir görsel değil.")
    except Exception as e:
        raise ValueError(f"Görsel açılırken hata oluştu: {str(e)}")


def convert_to_rgb(img: Image.Image) -> Image.Image:
    """
    Görseli RGB formatına dönüştürür.
    RGBA, grayscale, palette gibi farklı formatları normalize eder.
    
    Args:
        img: PIL Image (herhangi bir mod)
    
    Returns:
        RGB modunda PIL Image
    """
    if img.mode != "RGB":
        logger.debug(f"Görsel modu {img.mode}, RGB'ye dönüştürülüyor...")
        img = img.convert("RGB")
    return img


def resize_image(img: Image.Image, size: Tuple[int, int] = TARGET_SIZE) -> Image.Image:
    """
    Görseli hedef boyuta yeniden boyutlandırır.
    LANCZOS interpolasyonu kullanılır (en kaliteli algoritma).
    
    Args:
        img: PIL Image
        size: Hedef boyut (genişlik, yükseklik)
    
    Returns:
        Resize edilmiş PIL Image
    """
    if img.size != size:
        logger.debug(f"Görsel {img.size}'dan {size}'a resize ediliyor...")
        img = img.resize(size, Image.Resampling.LANCZOS)
    return img


def image_to_array(img: Image.Image) -> np.ndarray:
    """
    PIL Image'ı NumPy array'e dönüştürür.
    
    ÖNEMLİ: Değer aralığı [0-255] HAM olarak bırakılır.
    EfficientNetV2M built-in preprocessing içerdiği için ekstra
    normalization (0-1 veya -1+1) YAPILMAZ.
    
    Args:
        img: PIL Image (RGB)
    
    Returns:
        Shape: (H, W, 3), dtype: float32, range: [0-255]
    """
    return np.array(img, dtype=np.float32)


def add_batch_dimension(arr: np.ndarray) -> np.ndarray:
    """
    Tek bir görselin shape'ini batch formatına çevirir.
    Modelin beklediği shape: (batch_size, H, W, 3)
    
    Args:
        arr: Shape (H, W, 3) olan array
    
    Returns:
        Shape (1, H, W, 3) olan array
    """
    return np.expand_dims(arr, axis=0)


# ============================================================
# Ana Pipeline Fonksiyonu
# ============================================================

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    SCRUM-25: Ana görüntü işleme fonksiyonu.
    
    Gelen görsel byte verisini, modelin (EfficientNetV2M) bekledigi formata
    dönüştürür. Tüm preprocessing adımlarını tek noktada birleştirir.
    
    Pipeline:
        bytes → PIL Image → RGB → 224x224 → NumPy array → (1, 224, 224, 3)
    
    Args:
        image_bytes: Görsel dosyasının ham byte verisi
    
    Returns:
        Shape: (1, 224, 224, 3), dtype: float32, range: [0-255] (HAM)
        
    Raises:
        ValueError: Görsel açılamazsa, bozuksa veya işlenemezse
    """
    # 1. Bytes → PIL Image
    img = open_image_from_bytes(image_bytes)
    
    # 2. RGB'ye dönüştür
    img = convert_to_rgb(img)
    
    # 3. 224x224'e resize et
    img = resize_image(img, TARGET_SIZE)
    
    # 4. NumPy array'e çevir [0-255] HAM
    arr = image_to_array(img)
    
    # 5. Batch dimension ekle (1, 224, 224, 3)
    arr = add_batch_dimension(arr)
    
    return arr


def get_processor_info() -> dict:
    """
    Image processor hakkında bilgi döndürür.
    /health endpoint'inde kullanılabilir.
    
    Returns:
        Processor meta bilgileri
    """
    return {
        "target_size": list(TARGET_SIZE),
        "target_channels": TARGET_CHANNELS,
        "supported_formats": sorted(list(SUPPORTED_FORMATS)),
        "value_range": "[0-255] (raw, no normalization)",
        "interpolation": "LANCZOS",
    }


# ============================================================
# Test (doğrudan çalıştırılırsa)
# ============================================================
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
    
    print("\n" + "=" * 60)
    print("Image Processor Test")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("\nKullanım: python image_processor.py <görsel_dosya_yolu>")
        print("Örnek: python image_processor.py test.jpeg")
        print("\nProcessor bilgileri:")
        for key, value in get_processor_info().items():
            print(f"  {key}: {value}")
        sys.exit(1)
    
    image_path = sys.argv[1]
    print(f"\n📂 Görsel okunuyor: {image_path}")
    
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        print(f"✓ Dosya boyutu: {len(image_bytes) / 1024:.1f} KB")
        print(f"\n🔧 Preprocessing yapılıyor...")
        
        arr = preprocess_image(image_bytes)
        
        print(f"\n{'=' * 60}")
        print(f"  SONUÇ")
        print(f"{'=' * 60}")
        print(f"  Output shape : {arr.shape}")
        print(f"  Data type    : {arr.dtype}")
        print(f"  Min değer    : {arr.min():.2f}")
        print(f"  Max değer    : {arr.max():.2f}")
        print(f"  Ortalama     : {arr.mean():.2f}")
        print(f"  Aralık       : [0-255] (HAM, normalize edilmedi)")
        print(f"{'=' * 60}\n")
    
    except FileNotFoundError:
        print(f"\n❌ Dosya bulunamadı: {image_path}")
    except ValueError as e:
        print(f"\n❌ Görsel hatası: {e}")
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")