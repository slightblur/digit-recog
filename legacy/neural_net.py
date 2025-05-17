import numpy as np
from utils.matrix_ops import sigmoid, sigmoid_derivative, dot, subtract

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = dot(X, self.weights1) + self.bias1
        self.a1 = sigmoid(self.z1)
        self.z2 = dot(self.a1, self.weights2) + self.bias2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output, learning_rate=0.1):
        error = output - y 
        d_z2 = error * sigmoid_derivative(self.z2)
        d_w2 = dot(self.a1.T, d_z2)
        
        error_hidden = dot(d_z2, self.weights2.T)
        d_z1 = error_hidden * sigmoid_derivative(self.z1)
        d_w1 = dot(X.T, d_z1)
        
        self.weights1 -= learning_rate * d_w1
        self.bias1 -= learning_rate * np.sum(d_z1, axis=0, keepdims=True)
        self.weights2 -= learning_rate * d_w2
        self.bias2 -= learning_rate * np.sum(d_z2, axis=0, keepdims=True)

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}: Loss = {loss}")

    def predict(self, X):
        return self.forward(X)