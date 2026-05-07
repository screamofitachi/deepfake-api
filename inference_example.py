"""
EfficientNetV2M Deepfake Detector - Backend Inference Example
=============================================================
Model: best_efficientnetv2m.keras
Input: RGB image, [0-255], 224x224
Output: sigmoid 0-1 (>0.5 = Deepfake)
"""

import numpy as np
from PIL import Image
import tensorflow as tf

MODEL = tf.keras.models.load_model("best_efficientnetv2m.keras")
CLASS_NAMES = ["Real", "Deepfake"]
THRESHOLD = 0.5
INPUT_SIZE = (224, 224)


def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(INPUT_SIZE, Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict(image_path):
    x = preprocess(image_path)
    raw = MODEL.predict(x, verbose=0)[0][0]
    label_idx = 1 if raw > THRESHOLD else 0
    confidence = raw if label_idx == 1 else (1 - raw)
    return CLASS_NAMES[label_idx], float(confidence)


# Flask ornegi
"""
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/api/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400
    image = request.files["image"]
    image.save("/tmp/upload.jpg")
    label, confidence = predict("/tmp/upload.jpg")
    return jsonify({
        "label": label,
        "confidence": round(confidence, 4),
        "is_deepfake": label == "Deepfake"
    })
"""

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        label, conf = predict(sys.argv[1])
        print(f"{label} (confidence: {conf:.4f})")
    else:
        print("Kullanim: python inference_example.py <image_path>")
