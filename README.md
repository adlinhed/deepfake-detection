# Real and Fake Face Detection using Machine Learning

A deepfake detection web application that classifies facial images as real or deepfakes using transfer learning models. Then, integrated into an accessible Flask-based interface.

**Author:** Nur Adlin Fadhlina Binti Hedilisyam  
**Supervisor:** Dr. Abdullah Almasri  
**Institution:** Heriot-Watt University Malaysia — School of Mathematical and Computer Sciences   

---

## Models

All three models are trained on the [Yonsei University Real and Fake Face Detection Dataset](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection) using a two phase fine-tuning strategy with a shared augmentation pipeline and class weight balancing.

| Model       | Test Accuracy | Best Val Accuracy |
|-------------|--------------|-------------------|
| MobileNetV2 | 65.80%       | 68.63%            |
| ResNet50    | 64.50%       | 68.95%            |
| Xception    | 61.24%       | 61.76%            |

---

## Dataset

- **Source:** [CIPLAB @ Yonsei University](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection)
- **License:** CC BY-NC-SA 4.0 (non-commercial research use)
- **Size:** 2,041 images — 1,081 real, 960 fake
- **Fake subsets:** Easy, Mid, and Hard difficulty levels
- **Split:** 70% train / 15% validation / 15% test

---

## Project Structure

```
main/
├── data/
│   ├── test                        # Test set
│   ├── train                       # Training set
│   ├── val                         # Validation set
│   └── README.md                   
├── notebooks/
│   ├── dataset_preparation.ipynb   # Dataset splitting and validation
│   ├── mobilenetv2_training.ipynb  # MobileNetV2 training notebook
│   ├── resnet50_training.ipynb     # ResNet50 training notebook
│   ├── xception_training.ipynb     # Xception training notebook
│   └── README.md 
├── Web_app/
│   ├── app.py                      # Flask backend 
│   ├── models/
│   │   ├── README.md
│   │   ├── mobilenet_model.keras   # Trained MobileNetV2 model
│   │   ├── resnet_model.keras      # Trained ResNet50 model
│   │   └── xception_model.keras    # Trained Xception model
│   ├── static/
│   │   └── style.css               # Interface styling 
│   └── templates/
│        └── index.html             # Frontend template 
└── README.md

---

## Setup and Installation

### Prerequisites

- Python 3.9+
- pip

### Install dependencies

```bash
pip install flask tensorflow keras pillow numpy pandas scikit-learn matplotlib
```

### Run the application

```bash
python app.py
```

Then open your browser at `http://127.0.0.1:5000`.

> **Note:** All three models are loaded at server startup. Ensure the `models/` directory contains the `.keras` model files before running.

---

## How It Works

1. The user uploads a facial image (`.jpg`, `.png`, or `.jpeg`) from the "/data/test" images
2. The user selects a detection model (ResNet50, Xception, or MobileNetV2)
3. A confidence score and contextual explanation are returned and displayed alongside the image preview

---
