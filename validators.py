"""
File Validators - Deepfake Detection API
==========================================
Yüklenen dosyaların güvenli ve geçerli olduğunu doğrulayan
validasyon fonksiyonları.

Kabul kriterleri (SCRUM-4 epic):
- API, görsel haricindeki dosyaları (PDF, TXT vb.) reddetmeli
- Hatalı yüklemelerde 400 Bad Request gibi anlamlı hata kodları dönmeli

Tüm validasyon fonksiyonları, hata varsa HTTPException fırlatır.
Hata yoksa sessizce çalışır (None döner).
"""

import io
import logging
from typing import Optional

from fastapi import HTTPException, status
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)


# ============================================================
# Sabitler
# ============================================================

# Kabul edilen dosya uzantıları (lowercase)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Kabul edilen MIME types
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
}

# Dosya boyutu limitleri (byte cinsinden)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MIN_FILE_SIZE = 100  # 100 byte (çok küçük dosyalar bozuk olabilir)

# Görsel boyut limitleri (piksel)
MIN_IMAGE_DIMENSION = 32   # 32x32'den küçük görseller modelde anlamsız
MAX_IMAGE_DIMENSION = 8000  # Aşırı büyük görseller sunucuyu yorabilir


# ============================================================
# Validasyon Fonksiyonları
# ============================================================

def validate_filename(filename: Optional[str]) -> None:
    """
    Dosya adının uzantısını kontrol eder.
    
    Args:
        filename: Yüklenen dosyanın adı
    
    Raises:
        HTTPException 415: Geçersiz uzantı
        HTTPException 400: Dosya adı yok
    """
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dosya adı belirtilmemiş."
        )
    
    # Uzantıyı al (lowercase, son nokta sonrası)
    filename_lower = filename.lower()
    extension = "." + filename_lower.rsplit(".", 1)[-1] if "." in filename_lower else ""
    
    if extension not in ALLOWED_EXTENSIONS:
        allowed_list = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"'{extension}' uzantısı desteklenmiyor. "
                   f"Sadece şu formatlar kabul edilir: {allowed_list}"
        )


def validate_content_type(content_type: Optional[str]) -> None:
    """
    MIME type (Content-Type) kontrolü.
    
    Args:
        content_type: Dosyanın MIME type'ı
    
    Raises:
        HTTPException 415: Geçersiz MIME type
    """
    if not content_type:
        # Content-Type yoksa, uzantı kontrolü yeterli (varsayım: tarayıcı eklemiyor)
        logger.debug("Content-Type belirtilmemiş, sadece uzantı kontrolüyle devam.")
        return
    
    # "image/jpeg; charset=utf-8" gibi durumlar için sadece ana kısmı al
    main_type = content_type.split(";")[0].strip().lower()
    
    if main_type not in ALLOWED_MIME_TYPES:
        allowed_list = ", ".join(sorted(ALLOWED_MIME_TYPES))
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"İçerik tipi '{content_type}' desteklenmiyor. "
                   f"Sadece şu tipler kabul edilir: {allowed_list}"
        )


def validate_file_size(file_size: int) -> None:
    """
    Dosya boyutu kontrolü (minimum ve maksimum).
    
    Args:
        file_size: Byte cinsinden dosya boyutu
    
    Raises:
        HTTPException 413: Dosya çok büyük
        HTTPException 400: Dosya boş veya çok küçük
    """
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Yüklenen dosya boş."
        )
    
    if file_size < MIN_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dosya çok küçük ({file_size} byte). "
                   f"En az {MIN_FILE_SIZE} byte olmalı (geçerli bir görsel için)."
        )
    
    if file_size > MAX_FILE_SIZE:
        max_mb = MAX_FILE_SIZE / (1024 * 1024)
        actual_mb = file_size / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Dosya çok büyük ({actual_mb:.1f} MB). "
                   f"Maksimum {max_mb:.0f} MB kabul edilir."
        )


def validate_image_content(image_bytes: bytes) -> None:
    """
    Görselin gerçekten bir görsel olduğunu ve boyutlarının uygun olduğunu kontrol eder.
    
    Bu, "magic bytes" doğrulaması içerir — sadece uzantısı .jpg olan ama içeriği
    farklı olan dosyaları yakalar.
    
    Args:
        image_bytes: Görsel byte verisi
    
    Raises:
        HTTPException 400: Geçersiz veya bozuk görsel
        HTTPException 400: Görsel boyutu uygunsuz
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # Görsel verisini doğrula (bozuk mu?)
        
        # verify() sonrası tekrar açmak gerekir
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Yüklenen dosya geçerli bir görsel değil. "
                   "Lütfen JPG veya PNG formatında bir görsel yükleyin."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Görsel okunurken hata oluştu: {str(e)}"
        )
    
    # Boyut kontrolleri
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Görsel çok küçük ({width}x{height}). "
                   f"En az {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION} piksel olmalı."
        )
    
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Görsel çok büyük ({width}x{height}). "
                   f"En fazla {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION} piksel olmalı."
        )


# ============================================================
# Tüm Validasyonları Çalıştıran Ana Fonksiyon
# ============================================================

def validate_upload(filename: Optional[str], content_type: Optional[str], 
                    image_bytes: bytes) -> None:
    """
    Yüklenen dosyanın tüm validasyonlarını sırayla çalıştırır.
    
    Args:
        filename: Dosya adı
        content_type: MIME type
        image_bytes: Dosya içeriği (byte)
    
    Raises:
        HTTPException: Herhangi bir validasyon başarısız olursa
    """
    # Sırayla kontrol et (hızlıdan yavaşa)
    validate_filename(filename)              # En hızlı: string kontrolü
    validate_content_type(content_type)      # Hızlı: string kontrolü
    validate_file_size(len(image_bytes))     # Hızlı: int karşılaştırma
    validate_image_content(image_bytes)      # Yavaş: PIL ile görseli açar


def get_validation_info() -> dict:
    """
    Validasyon kurallarının özetini döndürür.
    /health veya dokümantasyon için kullanılabilir.
    """
    return {
        "allowed_extensions": sorted(list(ALLOWED_EXTENSIONS)),
        "allowed_mime_types": sorted(list(ALLOWED_MIME_TYPES)),
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
        "min_file_size_bytes": MIN_FILE_SIZE,
        "min_image_dimension": MIN_IMAGE_DIMENSION,
        "max_image_dimension": MAX_IMAGE_DIMENSION,
    }