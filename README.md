# 🧠 Brain Tumor Detection using Deep Learning (PyTorch)

## Overview
This project detects brain tumors using CT and MRI scans using a CNN built in PyTorch.

## Features
- CT / MRI classification
- Healthy / Tumor prediction
- Streamlit Web UI
- FastAPI Backend
- Real-time prediction

## Dataset
9618 images

Classes:
- CT Healthy
- CT Tumor
- MRI Healthy
- MRI Tumor

## Tech Stack
- Python
- PyTorch
- Streamlit
- FastAPI
- NumPy
- Matplotlib

## Results
Tumor Accuracy: ~96%

Modality Accuracy: 100%

## Run

Install:

pip install -r requirements.txt

Train:

python train.py

API:

uvicorn app.main:app --reload

Frontend:

streamlit run app/streamlit_app.py