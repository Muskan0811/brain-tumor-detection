from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = "models/model.keras"

# Load the new TF 2.15 compatible model
model = load_model(MODEL_PATH, compile=False)

IMAGE_SIZE = 128
CLASS_LABELS = sorted(os.listdir("models/classes")) if os.path.exists("models/classes") else ["glioma", "meningioma", "notumor", "pituitary"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No image uploaded"

    image = request.files["image"]
    path = "uploads/uploaded.jpg"
    image.save(path)

    img = load_img(path, target_size=(128,128))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    label = CLASS_LABELS[class_index]

    if label == "notumor":
        result = "No Tumor Detected"
    else:
        result = f"Tumor Detected: {label}"

    return render_template("index.html",
                           result=result,
                           confidence=f"{confidence*100:.2f}%")

if __name__ == "__main__":
    app.run(debug=True)
