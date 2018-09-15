from knn.calculate_distances import compute_big_matrix
from knn.calculate_distances import save
from load_file import load_cifar10

Xtr, Ytr, Xte, Yte = load_cifar10('data/cifar10/')
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)  # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)  # Xte_rows becomes 10000 x 3072

big_matrix = compute_big_matrix(Xtr_rows, Xte_rows)
save(big_matrix, 'train_test_distances.pkl')
