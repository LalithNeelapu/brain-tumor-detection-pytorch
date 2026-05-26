import torch
from torchvision import transforms
from model.cnn_model import CNNModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

model = CNNModel().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()


def predict_image(image):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.sigmoid(output).cpu().numpy().flatten()

    pred_modality = "MRI" if probs[0] > 0.5 else "CT"
    pred_tumor = "Tumor" if probs[1] > 0.5 else "Healthy"

    return pred_modality, pred_tumor, probs