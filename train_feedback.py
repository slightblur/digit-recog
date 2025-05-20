import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

# Paths
IMAGE_DIR = "data/images"
CSV_FILE = "data/labels.csv"
MODEL_PATH = "models/cnn_mnist_model.h5"

def load_feedback_data():
    images = []
    labels = []

    if not os.path.exists(CSV_FILE):
        print("No feedback data found.")
        return np.array([]), np.array([])

    with open(CSV_FILE, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 2:
                continue
            filename, label = row
            img_path = os.path.join(IMAGE_DIR, filename)
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))  # Ensure image size consistency
            img = img.astype("float32") / 255.0
            images.append(img)
            labels.append(int(label))

    X = np.expand_dims(np.array(images), -1)
    y = np.array(labels)
    return X, y

def train_on_feedback():
    X, y = load_feedback_data()
    if len(X) == 0:
        print("No new feedback to train on.")
        return

    print("✅ Feedback data loaded.")
    print(f"🖼️ Training on {len(X)} samples...")

    model = load_model(MODEL_PATH)

    # 🔧 Recompile model to ensure training configuration is set
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(X, y, epochs=10, batch_size=16)

    model.save(MODEL_PATH)
    print("✅ Model updated with feedback data.")

if __name__ == "__main__":
    train_on_feedback()