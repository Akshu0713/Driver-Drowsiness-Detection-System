# 🚗 Driver Drowsiness Detection System using Machine Learning

This project implements a real-time Driver Drowsiness Detection System using **deep learning** and **computer vision** to prevent road accidents caused by driver fatigue. By monitoring the driver's eye state (open or closed), the system triggers alerts when signs of drowsiness are detected.

---

## 📌 Features

- 🎯 Real-time drowsiness detection using webcam
- 👁️ Eye state classification (Open/Closed)
- 🧠 CNN model with **Transfer Learning (InceptionV3)**
- 🎥 Face and eye detection using **OpenCV Haar Cascade** and **Dlib**
- 🔊 Audio alerts for drowsy state
- 📊 Achieves **92%+ accuracy** on diverse conditions (glasses, lighting, etc.)

---

## 📁 Dataset

**MRL Eye Dataset** – An open-source dataset containing over **80,000 images** of eyes under various conditions:
- With/without glasses
- Different lighting environments
- Varying reflections and angles

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- Dlib
- NumPy, Matplotlib
- Google Colab / Jupyter Notebook

---

## ⚙️ Model Architecture

- **Base Model:** InceptionV3 (pre-trained on ImageNet)
- **Custom Layers:** Flatten → Dense → Dropout → Dense (Binary Classification)
- **Detection Pipeline:**
  - Capture frames from webcam
  - Detect face and eyes
  - Predict eye state using trained model
  - Trigger alert if eyes are closed beyond threshold time

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/yourusername/driver-drowsiness-detection.git
cd driver-drowsiness-detection

# Install dependencies
pip install -r requirements.txt

# Run the detection script
python detect_drowsiness.py
