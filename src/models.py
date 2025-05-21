import numpy as np

class LogisticRegressionGD:
    def __init__(self, lr=0.01, epochs=1000, lam=0.0, class_weight=None):
        self.lr = lr
        self.epochs = epochs
        self.lam = lam
        self.class_weight = class_weight or {0: 1.0, 1: 1.0}

    @staticmethod
    def _sigmoid(z):
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        neg = ~pos
        out[pos] = 1 / (1 + np.exp(-z[pos]))
        exp_z = np.exp(z[neg])
        out[neg] = exp_z / (1 + exp_z)
        return out

    def fit(self, X, y):
        m, n = X.shape
        self.W = np.zeros(n)
        self.b = 0.0
        # gradient descent with class weights
        for _ in range(self.epochs):
            z = X.dot(self.W) + self.b
            p = self._sigmoid(z)
            # compute sample weights
            sample_w = np.array([self.class_weight[label] for label in y])
            # error term
            error = p - y
            # weighted gradients
            grad_W = (X.T.dot(sample_w * error)) / m + 2 * self.lam * self.W
            grad_b = np.sum(sample_w * error) / m
            # parameter update
            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b
        return self

    def predict(self, X):
        probs = self._sigmoid(X.dot(self.W) + self.b)
        return (probs >= 0.5).astype(int)



class Perceptron:
    def __init__(self, lr=0.01, epochs=100, class_weight=None):
        self.lr = lr
        self.epochs = epochs
        self.class_weight = class_weight or {0: 1.0, 1: 1.0}

    def fit(self, X, y):
        m, n = X.shape
        self.W = np.zeros(n)
        self.b = 0.0
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                pred = 1 if xi.dot(self.W) + self.b > 0 else 0
                update = yi - pred
                # weight update by class
                w = self.class_weight.get(yi, 1.0)
                self.W += self.lr * w * update * xi
                self.b += self.lr * w * update
        return self

    def predict(self, X):
        return (X.dot(self.W) + self.b > 0).astype(int)


class LinearSVM:
    def __init__(self, lr=0.001, epochs=1000, C=1.0, class_weight=None):
        self.lr = lr
        self.epochs = epochs
        self.C = C
        self.class_weight = class_weight or {0: 1.0, 1: 1.0}

    def fit(self, X, y):
        # convert labels {0,1} to {-1,1}
        y_mod = np.where(y == 1, 1, -1)
        m, n = X.shape
        self.W = np.zeros(n)
        self.b = 0.0
        for _ in range(self.epochs):
            for xi, yi_mod, yi_orig in zip(X, y_mod, y):
                margin = yi_mod * (xi.dot(self.W) + self.b)
                w = self.class_weight.get(yi_orig, 1.0)
                if margin < 1:
                    dW = -yi_mod * xi * w + 2 * self.C * self.W
                    db = -yi_mod * w
                else:
                    dW = 2 * self.C * self.W
                    db = 0.0
                self.W -= self.lr * dW
                self.b -= self.lr * db
        return self

    def predict(self, X):
        return (X.dot(self.W) + self.b >= 0).astype(int)
