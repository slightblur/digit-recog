import tensorflow as tf
from tensorflow.python.keras import layers, models
from tensorflow.keras.datasets import mnist

def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    model = create_cnn_model()
    model.summary()

    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    model.save('models/cnn_mnist_model.h5')
    print("✅ Model trained and saved to models/cnn_mnist_model.h5")

if __name__ == '__main__':
    main()