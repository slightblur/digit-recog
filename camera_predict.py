import cv2
import numpy as np
import tensorflow as tf
from utils.image_utils import preprocess_image
from feedback_handler import save_feedback_image

def main():
    model = tf.keras.models.load_model('models/cnn_mnist_model.h5')
    print("✅ Model loaded. Press SPACE to predict, ESC to exit.")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Show live camera feed
        cv2.imshow("Draw a digit and press SPACE", frame)

        key = cv2.waitKey(1)

        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            processed = preprocess_image(frame)
            prediction = model.predict(processed)
            predicted_digit = np.argmax(prediction)

            print(f"🔢 Predicted digit: {predicted_digit}")

            save_feedback_image((processed.squeeze() * 255).astype("uint8"), predicted_digit)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()