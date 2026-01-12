import numpy as np

# Training data (2D features)
X_train = np.array([
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2],
    [2, 3],
    [3, 2],
    [3, 3],
    [4, 3],
    [4, 4],
    [5, 4],
    [5, 5],
    [6, 5],
    [6, 6],
    [7, 6],
    [7, 7],
    [8, 7],
    [8, 8],
    [9, 8]
])

# Labels (0 = Class A, 1 = Class B)
y_train = np.array([
    0, 0, 0, 0, 0, 0, 0,
    0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1
])

# Test data
X_test = np.array([
    [2, 2],
    [3, 3],
    [4, 4],
    [6, 6],
    [7, 7]
])
y_test = np.array([
    0, 
    0,  
    0,  
    1, 
    1   
])