import cv2
import numpy as np

def preprocess_image(img):
    """
    Preprocesses a webcam frame to extract and normalize a digit for prediction.
    Steps:
    - Convert to grayscale
    - Apply Gaussian blur and thresholding
    - Find contours
    - Crop the largest contour
    - Resize to 28x28
    - Invert colors (to match MNIST: white digit on black)
    - Normalize pixel values to [0,1]
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    digit = thresh[y:y+h, x:x+w]

    resized = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)

    # Invert image to match MNIST (white digit on black background)
    inverted = 255 - resized

    # Normalize
    normalized = inverted.astype(np.float32) / 255.0

    # Debug view
    cv2.imshow("Preprocessed Digit", cv2.resize(inverted, (280, 280), interpolation=cv2.INTER_NEAREST))

    return normalized.reshape(1, 28, 28, 1)