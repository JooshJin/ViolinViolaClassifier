import numpy as np

class LogisticRegressionGD:
    def __init__(self, lr=0.01, epochs=500, lam=0.0):
        self.lr, self.epochs, self.lam = lr, epochs, lam

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        self.W = np.zeros(n)
        self.b = 0.0
        for _ in range(self.epochs):
            z = X.dot(self.W) + self.b
            a = self._sigmoid(z)
            dz = a - y
            dW = (X.T.dot(dz) + self.lam * self.W) / m
            db = dz.sum() / m
            self.W -= self.lr * dW
            self.b -= self.lr * db
        return self

    def predict(self, X, thresh=0.5):
        return (self._sigmoid(X.dot(self.W) + self.b) >= thresh).astype(int)


class Perceptron:
    def __init__(self, lr=0.01, epochs=100):
        self.lr, self.epochs = lr, epochs

    def fit(self, X, y):
        m, n = X.shape
        self.W = np.zeros(n)
        self.b = 0.0
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                pred = 1 if xi.dot(self.W) + self.b > 0 else 0
                update = yi - pred
                self.W += self.lr * update * xi
                self.b += self.lr * update
        return self

    def predict(self, X):
        return (X.dot(self.W) + self.b > 0).astype(int)


class LinearSVM:
    def __init__(self, lr=0.001, epochs=1000, C=1.0):
        self.lr, self.epochs, self.C = lr, epochs, C

    def fit(self, X, y):
        # convert labels {0,1} to {-1,1}
        y_mod = np.where(y == 1, 1, -1)
        m, n = X.shape
        self.W = np.zeros(n)
        self.b = 0.0
        for _ in range(self.epochs):
            for xi, yi in zip(X, y_mod):
                margin = yi * (xi.dot(self.W) + self.b)
                if margin < 1:
                    dW = -yi * xi + 2 * self.C * self.W
                    db = -yi
                else:
                    dW = 2 * self.C * self.W
                    db = 0
                self.W -= self.lr * dW
                self.b -= self.lr * db
        return self

    def predict(self, X):
        return (X.dot(self.W) + self.b >= 0).astype(int)
