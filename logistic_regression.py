import numpy as np

class LogisticRegressionScratch:
    def __init__(self, learning_rate = 0.1, n_iter = 1000, verbose= False):
        self.bias = None
        self.weights = None
        self.lr = learning_rate
        self.n_iter = n_iter
        self.verbose = verbose
        self.losses = []

    def sigmoid(self, z):
        return (1/(1 + np.exp(-z)))

    def compute_loss(self, y, y_pred):
        # numerical stability
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)

        loss = -(1 / len(y)) * np.sum(
            y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
        )
        return loss

    def fit(self, X, y):
            # Initialize
            m,n = X.shape
            self.bias = 0
            self.weights = np.zeros(n)

            for i in range (self.n_iter):
                # Forward pass
                z = X @ self.weights + self.bias
                y_pred = self.sigmoid(z)

                # Calc gradient
                db = (1/m) * np.sum(y_pred - y)
                dw = (1/m) * (X.T @ (y_pred -y))

                # Convergence theorem - Update
                self.bias -= self.lr * db
                self.weights -= self.lr * dw

                # Loss tracking
                loss = self.compute_loss(y, y_pred)
                self.losses.append(loss)

                if self.verbose and i % 100 == 0:
                    print(f"Iteration {i}, Loss: {loss:.4f}")
                
    def get_probabilities(self, X):
        z = self.bias + (X @ self.weights)
        return self.sigmoid(z) # returning probabilities -> 0.89,0.34
        
    def predict(self, X, threshold = 0.5):
        probabilities = self.get_probabilities(X)
        y_pred_bool = probabilities >= threshold
        return y_pred_bool.astype(int)
        
    
    