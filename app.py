from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import io
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

app = Flask(__name__)


# 모델 로드
model = tf.keras.models.load_model("model.keras")
class_names = np.load("class_names.npy", allow_pickle=True)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((300, 300))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image_bytes = file.read()
    input_tensor = preprocess_image(image_bytes)

    prediction = model.predict(input_tensor)
    pred_idx = np.argmax(prediction[0])
    label = class_names[pred_idx]
    confidence = float(prediction[0][pred_idx])

    return jsonify({"label": str(label), "confidence": confidence})

@app.route("/", methods=["GET"])
def home():
    return "🚀 AI Facility Classifier API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
