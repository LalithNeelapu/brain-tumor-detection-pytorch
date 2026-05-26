import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI, UploadFile, File
from PIL import Image
from utils.predict import predict_image

app = FastAPI()


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")

    modality, tumor, probs = predict_image(image)

    modality_conf = probs[0] if modality == "MRI" else (1 - probs[0])
    tumor_conf = probs[1] if tumor == "Tumor" else (1 - probs[1])

    return {
    "modality": modality,
    "tumor": tumor,
    "modality_confidence": f"{modality_conf*100:.2f}%",
    "tumor_confidence": f"{tumor_conf*100:.2f}%"
}