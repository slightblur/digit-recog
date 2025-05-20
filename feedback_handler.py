import os
import uuid
import csv
import cv2

DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
CSV_FILE = os.path.join(DATA_DIR, "labels.csv")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def save_feedback_image(image, predicted_label):
    # Ask for feedback
    feedback = input("Is this prediction correct? (y/n): ").strip().lower()
    
    if feedback == "y":
        return  # No need to store if correct

    correct_digit = input("What was the correct digit? (0-9): ").strip()
    
    if not correct_digit.isdigit() or not (0 <= int(correct_digit) <= 9):
        print("Invalid digit. Feedback discarded.")
        return

    # Save image
    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join(IMAGE_DIR, filename)
    cv2.imwrite(filepath, image)

    # Append to CSV
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([filename, correct_digit])

    print(f"Saved incorrect prediction for retraining: {filename} → {correct_digit}")