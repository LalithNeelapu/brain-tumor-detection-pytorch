from PIL import Image
from utils.predict import predict_image

image_path = r"C:\Users\lalit\Desktop\BTD-CT_MRI\dataset\DL Project Dataset\Brain Tumor CT scan Images\Tumor\ct_tumor (4).png"

img = Image.open(image_path).convert("RGB")

modality, tumor, probs = predict_image(img)

modality_conf = probs[0] if modality == "MRI" else (1 - probs[0])
tumor_conf = probs[1] if tumor == "Tumor" else (1 - probs[1])

print("\nPrediction Result")
print("-------------------")
print(f"Predicted Modality: {modality}")
print(f"Predicted Tumor Status: {tumor}")
print(f"Prediction Confidence (Modality): {modality_conf*100:.2f}%")
print(f"Prediction Confidence (Tumor): {tumor_conf*100:.2f}%")