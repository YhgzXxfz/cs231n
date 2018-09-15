import numpy as np
from knn.k_nearest_neighbor_optimized import KNearestNeighbor

from knn.calculate_distances import load
from knn.calculate_distances import save
from load_file import load_cifar10

Xtr, Ytr, Xte, Yte = load_cifar10('data/cifar10/')
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)  # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)  # Xte_rows becomes 10000 x 3072

Xtr_extend = np.vstack((Xtr_rows, Xtr_rows))
Ytr_extend = np.array(Ytr.tolist() + Ytr.tolist())
val_size = 10000


validation_accuracies = {}
for fold in range(5):
    print("FOLD "+str(fold))
    Xval_rows = Xtr_extend[fold * val_size:(fold+1) * val_size, :]
    Yval = Ytr_extend[fold * val_size:(fold+1) * val_size]
    Xtr_rows = Xtr_extend[(fold+1) * val_size:(fold+5)*val_size, :]
    Ytr = Ytr_extend[(fold+1) * val_size:(fold+5)*val_size]

    # ori = load('train_val_fold_'+str(3)+'_distances.pkl')
    ori = load('train_val_fold_'+str(fold)+'_distances.pkl')
    print("load finished")

    temp = ori
    for i in range(val_size):
        doubled = temp[i].tolist()
        doubled.extend(doubled)
        # doubled = temp[i].extend(temp[i])
        temp[i] = doubled[(fold+1)*val_size:(fold+5)*val_size]

    distance_matrix = temp

    knn = KNearestNeighbor(distance_matrix)

    knn.train(Xtr_rows, Ytr)
    for k in [1, 2, 3]:
        Yval_predict = knn.predict(Xval_rows, degree=2, k=k)
        acc = np.mean(Yval_predict == Yval)
        print('k: %d, accuracy: %f' % (k, acc,))

        if k not in validation_accuracies:
            validation_accuracies[k] = []
        validation_accuracies[k].append(acc)

save(validation_accuracies, 'validation_accuracies')
