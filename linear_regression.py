import numpy as np

# Linear Regression - Scratch Implementation
class LinearRegressionScratch:
    def __init__(self, learning_rate = 0.01, n_iter = 1000, verbose=False):
        self.bias = None
        self.weights = None
        self.lr = learning_rate
        self.n_iter = n_iter
        self.losses = []
        self.verbose = verbose

   
    def fit(self, X, y ): # X_train, y_train
        m,n = X.shape # nos of samples , features
        y = y.reshape(-1)
        
         # Step1 : Initialize params
        self.bias = 0
        self.weights = np.zeros(n) 

        # Gradient Descent
        for i in range (self.n_iter):
            
            # Step2 : Calculate y_pred
            y_pred = self.bias + (X @ self.weights)

            # Step3 : Calculate Gradient
            db = (1/m) * np.sum(y_pred - y)
            dw = (1/m) * (X.T @ (y_pred - y))

            # Step4 : Convergence Theorem
            self.bias -= self.lr * db
            self.weights -= self.lr * dw
            
            # Losses
            loss = (1/(2*m)) * np.sum((y_pred - y)**2)
            self.losses.append(loss)

            # Debugging support
            if self.verbose and i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss:.4f}")


    def predict(self, X):
        y_pred = (X @ self.weights) + self.bias
        return y_pred

    def r2_score(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)

