import time

import numpy as np
from knn.k_nearest_neighbor_optimized import KNearestNeighbor

from knn.calculate_distances import load
from load_file import load_cifar10

Xtr, Ytr, Xte, Yte = load_cifar10('data/cifar10/')
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)  # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)  # Xte_rows becomes 10000 x 3072

Xte_rows_small = Xte_rows[0:100, :]
Yte_small = Yte[0:100]

## RUN NEAREST NEIGHBOR
# nn = NearestNeighbor()  # create a Nearest Neighbor classifier class
# nn.train(Xtr_rows, Ytr)  # train the classifier on the training images and labels
# Yte_predict = nn.predict(Xte_rows_small, degree=2)  # predict labels on the test images
# # and now print the classification accuracy, which is the average number
# # of examples that are correctly predicted (i.e. label matches)
# print('accuracy: %f' % (np.mean(Yte_predict == Yte_small)))

## RUN K NEAREST NEIGHBOR
# knn = KNearestNeighbor(k=5)
# knn.train(Xtr_rows, Ytr)
# Yte_predict = knn.predict(Xte_rows_small, degree=2)
# print('accuracy: %f' % (np.mean(Yte_predict == Yte_small)))

## RUN K NEAREST NEIGHBOR OPTIMIZED
print("BEGIN LOADING")
print(time.time())
distance_matrix = load('train_test_distances.pkl')
print("END LOADING")
print(time.time())

knn_opt = KNearestNeighbor(distance_matrix)
knn_opt.train(Xtr_rows, Ytr)
print("BEGIN PREDICTION")
print(time.time())
Yte_predict = knn_opt.predict(Xte_rows, degree=2, k=1)
print("END PREDICTION")
print(time.time())

print(Yte_predict.shape)
print('accuracy: %f' % (np.mean(Yte_predict == Yte)))

