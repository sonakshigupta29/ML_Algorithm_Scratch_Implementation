import numpy as np

def create_sample_data():
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X[:, 0] + np.random.randn(100)
    return X, y

#This follows: 𝑦 = 3𝑥 + 4 + 𝑛𝑜𝑖𝑠𝑒 -> Perfect for validation.