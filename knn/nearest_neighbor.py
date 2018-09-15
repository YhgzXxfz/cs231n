import numpy as np


class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, Y):
        self.Xtr = X
        self.Ytr = Y

    def predict(self, X, degree=1):
        num_test = X.shape[0]

        Ypred = np.zeros(num_test, dtype=self.Ytr.dtype)

        for i in range(num_test):
            print(i)
            if degree == 1:
                distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            else:
                distances = np.power(np.sum(np.power(self.Xtr-X[i, :], degree), axis=1), 1/degree)
            min_index = np.argmin(distances)
            Ypred[i] = self.Ytr[min_index]

        return Ypred
