import torch
from train import model, test_loader
from sklearn.metrics import (
    classification_report,
    accuracy_score
)

model.eval()

y_true = []
y_pred = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to("cuda" if torch.cuda.is_available() else "cpu")

        outputs = model(images)

        preds = (
            torch.sigmoid(outputs)
            > 0.5
        ).cpu().numpy()

        y_pred.extend(preds[:,1])

        y_true.extend(
            labels[:,1].numpy()
        )

print(
    classification_report(
        y_true,
        y_pred
    )
)

print(
    "Accuracy:",
    accuracy_score(
        y_true,
        y_pred
    )
)