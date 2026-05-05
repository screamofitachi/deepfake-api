from fastapi import FastAPI

# FastAPI uygulamasını oluştur
app = FastAPI(
    title="Deepfake Detection API",
    description="Yüklenen görselin gerçek mi yoksa deepfake mi olduğunu tespit eden API",
    version="0.1.0"
)


# Ana sayfa endpoint'i
@app.get("/")
def root():
    return {
        "message": "Deepfake Detection API çalışıyor!",
        "status": "active",
        "version": "0.1.0"
    }


# Sağlık kontrolü endpoint'i (API ayakta mı diye kontrol için)
@app.get("/health")
def health_check():
    return {"status": "ok"}