import numpy as np

from knn.k_nearest_neighbor import KNearestNeighbor
from load_file import load_cifar10

Xtr, Ytr, Xte, Yte = load_cifar10('data/cifar10/')
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)  # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)  # Xte_rows becomes 10000 x 3072

val_size = 100
Xval_rows = Xtr_rows[:val_size, :]
Yval = Ytr[:val_size]
Xtr_rows = Xtr_rows[val_size:, :]
Ytr = Ytr[val_size:]

validation_accuracies = []
for k in [1, 3, 5, 10, 20, 50, 100]:
    knn = KNearestNeighbor(k)
    knn.train(Xtr_rows, Ytr)
    Yval_predict = knn.predict(Xval_rows, degree=2)
    acc = np.mean(Yval_predict == Yval)
    print('accuracy: %f' % (acc,))

    validation_accuracies.append((k, acc))
