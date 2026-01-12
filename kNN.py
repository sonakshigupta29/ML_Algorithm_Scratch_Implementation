import numpy as np

# KNN Implementation from scratch - for binary classification

class KNNClassifier:
    def __init__(self, k=3):
        # Initialize k ie. nos of nearest neighbors
        self.k = k
    
    def _euclidean_dist(self, x1, x2):
        # Calculate Euclidean distance between two points 
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def fit(self, X, y): 
        # Store training data (KNN is a lazy learner)
        self.X_train = X
        self.y_train = y

    def _predict_one(self, x):
        # Distance of input point x from all training points -> [1,3,5,2]
        distances = [self._euclidean_dist(x, x_train) for x_train in self.X_train]
        
        # Sorted indices of k smallest distances (nearest neighbors) -> [1,4,2,3]
        knn_indices = np.argsort(distances)[:self.k]
        
        # Class labels at that index of k nearest neighbors -> [1,2,3,5]
        knn_classes = [self.y_train[i] for i in knn_indices]
        
        # Performing majority voting to decide final class-> answer
        majority_class = np.argmax(np.bincount(knn_classes))
        return majority_class

    def predict(self, X):
        y_pred = [self._predict_one(x) for x in X]
        return np.array(y_pred)

