import numpy as np


def load_sample_data():
    """
    Features:
    ---------
    X[:,0] → hours studied
    X[:,1] → number of practice tests taken

    Label:
    ------
    y = 1 → Pass
    y = 0 → Fail
    added some noisiness in data
    """

    X = np.array([
        [1, 1],
        [1, 2],
        [2, 1],
        [2, 2],
        [2, 3],
        [3, 2],
        [3, 3],
        [3, 4],   # borderline
        [4, 3],
        [4, 4],
        [4, 5],
        [5, 4],   # noise (failed despite effort)
        [5, 5],
        [6, 5],
        [6, 6],
        [7, 6],
        [7, 7],
        [8, 7],
        [8, 8]
    ], dtype=float)

    y = np.array([
        0,  # 1,1
        0,  # 1,2
        0,  # 2,1
        0,  # 2,2
        0,  # 2,3
        0,  # 3,2
        0,  # 3,3
        1,  # 3,4
        1,  # 4,3
        1,  # 4,4
        1,  # 4,5
        0,  # 5,4  <-- noise
        1,  # 5,5
        1,  # 6,5
        1,  # 6,6
        1,  # 7,6
        1,  # 7,7
        1,  # 8,7
        1   # 8,8
    ], dtype=int)

    return X, y
