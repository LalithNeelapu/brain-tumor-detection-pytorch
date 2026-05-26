import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from model.cnn_model import CNNModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_root = r"dataset\DL Project Dataset"

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        for modality_folder in os.listdir(root_dir):
            modality_path = os.path.join(root_dir, modality_folder)
            if not os.path.isdir(modality_path):
                continue

            lower_mod = modality_folder.lower()
            if "ct" in lower_mod:
                modality_label = 0
            elif "mri" in lower_mod:
                modality_label = 1
            else:
                continue

            for status_folder in os.listdir(modality_path):
                status_path = os.path.join(modality_path, status_folder)
                if not os.path.isdir(status_path):
                    continue

                lower_stat = status_folder.lower()
                if "healthy" in lower_stat:
                    tumor_label = 0
                elif "tumor" in lower_stat:
                    tumor_label = 1
                else:
                    continue

                for img_file in os.listdir(status_path):
                    if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.image_paths.append(os.path.join(status_path, img_file))
                        self.labels.append([modality_label, tumor_label])

        print("Total images found:", len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


dataset = BrainTumorDataset(dataset_root, transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

model = CNNModel().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        modality_loss = criterion(outputs[:, 0], labels[:, 0])
        tumor_loss = criterion(outputs[:, 1], labels[:, 1])

        loss = modality_loss + tumor_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "model.pth")
print("Model saved!")

# ---------------- EVALUATION ----------------

model.eval()
y_true_modality, y_true_tumor = [], []
y_pred_modality, y_pred_tumor = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = torch.sigmoid(model(images)).cpu().numpy()
        preds = (outputs > 0.5).astype(int)

        y_true_modality.extend(labels[:, 0].numpy())
        y_true_tumor.extend(labels[:, 1].numpy())
        y_pred_modality.extend(preds[:, 0])
        y_pred_tumor.extend(preds[:, 1])

print("\nTumor Classification Report:")
print(classification_report(y_true_tumor, y_pred_tumor))

print("\nModality Classification Report:")
print(classification_report(y_true_modality, y_pred_modality))