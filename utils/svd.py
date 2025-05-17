import numpy as np

def compute_svd(matrix):
    U, S, Vt = np.linalg.svd(matrix)
    return U, S, Vt

def compute_eigen(matrix):
    values, vectors = np.linalg.eig(matrix)
    return values, vectors