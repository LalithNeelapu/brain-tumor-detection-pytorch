# 🧠 AI-Based Brain Tumor Detection using Deep Learning (PyTorch)

An end-to-end Deep Learning application that detects brain tumors from CT and MRI scans using a Convolutional Neural Network (CNN) built with PyTorch.

The system predicts:

✅ Imaging Modality → CT / MRI  
✅ Tumor Status → Healthy / Tumor  

The project includes:

- Model Training
- Evaluation Pipeline
- Real-Time Prediction
- FastAPI Backend
- Streamlit Web Interface

---

# 📌 Project Overview

This application processes medical imaging data and performs multi-task classification.

Input:

Medical Image (CT / MRI)

Output:

1. Scan Type
   - CT
   - MRI

2. Tumor Status
   - Healthy
   - Tumor

The complete pipeline includes preprocessing, training, evaluation, inference, and deployment.

---

# 🚀 Features

- Brain Tumor Detection
- CT / MRI Classification
- Confidence Score Prediction
- FastAPI REST API
- Streamlit Interactive UI
- End-to-End Deployment
- Real-Time Inference

---

# 📂 Dataset

Dataset Size:

9618 Images

Classes:

- CT Healthy
- CT Tumor
- MRI Healthy
- MRI Tumor

Split:

- Train → 80%
- Test → 20%

---

# 🏗 Model Architecture

Model:

CNN (Convolutional Neural Network)

Input:

64 × 64 RGB Image

Architecture:

Conv → ReLU → MaxPool  
Conv → ReLU → MaxPool  
Flatten  
Dense Layers

Output:

[ Modality , Tumor ]

---

# 🛠 Tech Stack

- Python
- PyTorch
- FastAPI
- Streamlit
- NumPy
- Matplotlib
- PIL

---

# 📈 Evaluation Results

## Tumor Detection

| Metric | Score |
|--------|------|
| Accuracy | 98.13% |
| Precision | 98% |
| Recall | 98% |
| F1 Score | 98% |

---

## Modality Classification

| Metric | Score |
|--------|------|
| Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1 Score | 100% |

---

# 🖼 Screenshots

## Streamlit Interface

![Streamlit UI](screenshots/streamlit_ui.png)

---

## FastAPI Backend

![FastAPI](screenshots/fastapi_docs.png)

---

## Prediction Example

![Prediction](screenshots/prediction_result.png)

---

# ⚙ Installation

Clone Repository

```bash
git clone https://github.com/LalithNeelapu/brain-tumor-detection-pytorch.git
```

Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Run Project

Train Model

```bash
python train.py
```

Evaluate Model

```bash
python evaluate.py
```

Run FastAPI

```bash
uvicorn app.main:app --reload
```

Run Streamlit

```bash
streamlit run app/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

---

# 📁 Project Structure

```text
BTD-CT_MRI
│
├── app
├── model
├── utils
├── screenshots
├── train.py
├── evaluate.py
├── test_predict.py
├── requirements.txt
├── README.md
```

---

# 🔮 Future Improvements

- Transfer Learning
- Explainable AI (Grad-CAM)
- Cloud Deployment
- Better Generalization
- Multi-class Tumor Classification

---

# ⚠ Disclaimer

This project is developed for educational and research purposes only and is not intended for medical diagnosis.