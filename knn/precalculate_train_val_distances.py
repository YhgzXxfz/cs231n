import time

from knn.calculate_distances import compute_big_matrix
from knn.calculate_distances import save
from load_file import load_cifar10

Xtr, Ytr, Xte, Yte = load_cifar10('data/cifar10/')
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)  # Xtr_rows becomes 50000 x 3072

folds = 5
batch_size = int(Xtr.shape[0] / folds)

print(time.time())
# for i in range(folds):
#     print("FOLD: " + str(i))
#     big_matrix = compute_big_matrix(Xtr_rows, Xtr_rows[i:i+batch_size, :])
#     save(big_matrix, 'train_val_fold_' + str(i) + '_distances.pkl')
#     print(len(big_matrix))

for i in range(1):
    i = 4
    print("FOLD: " + str(i))
    big_matrix = compute_big_matrix(Xtr_rows, Xtr_rows[i:i+batch_size, :])
    save(big_matrix, 'train_val_fold_' + str(i) + '_distances.pkl')
    print(len(big_matrix))

print(time.time())
