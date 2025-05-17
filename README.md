# ✍️ Handwritten Digit Recognition with CNN & Webcam (uv-managed)

A real-time handwritten digit recognition app using a Convolutional Neural Network (CNN) trained on the MNIST dataset, with live digit capture via your webcam.  

---

## 📸 Features

- 🧠 Trained TensorFlow CNN model for MNIST digits (0–9)
- 🎥 Live webcam input to read handwritten digits
- 🔍 Digit preprocessing to match MNIST format
- ⚡ Fast and clean Python project management with `uv`
- 🪞 Preview what the model "sees" before prediction

---

## 🚀 Getting Started

### 1. ✅ Prerequisites

- Python 3.8+ installed
- `uv` installed:  
  [Install uv →](https://github.com/astral-sh/uv#installation)

---

### 2. 📦 Install Dependencies with `uv`

```
uv venv         # Create a virtual environment
uv sync         # Install all dependencies from pyproject.toml + uv.lock
```

---

### 3. 🧠 Train the Model (only once)

```
uv run main.py
```

* Trains a CNN on MNIST
* Saves the model to `models/cnn_mnist_model.h5`

---

### 4. 📷 Predict Digits Using Webcam

```
uv run camera_predict.py
```

* Press `SPACE` to predict a digit
* Press `ESC` to quit
* A popup shows the preprocessed image
* The prediction is printed in the terminal

---

## 📌 Tips for Better Results

* Write bold, large digits on clean white paper
* Hold the paper upright and centered in the webcam
* Ensure good lighting (avoid shadows)
* Use simple, MNIST-like digits

---

## 🐞 Found a Bug or Issue?

If you encounter a bug, unexpected behavior, or have a suggestion — please feel free to reach out.

I'm new to neural networks and programming in general so your feedback is **greatly appreciated** as it'll help me improve 