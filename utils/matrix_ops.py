import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def dot(a, b):
    return np.dot(a, b)

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b