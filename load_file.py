import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_cifar10(path):
    Xtr = np.array([])
    Ytr = np.array([])
    for i in range(1, 6):
        filename = 'data_batch_' + str(i)
        dict = unpickle(path + filename)
        curr_x, curr_y = dict[b'data'], dict[b'labels']
        if Xtr.size == 0:
            Xtr = curr_x
            Ytr = curr_y
        else:
            Xtr = np.vstack((Xtr, curr_x))
            Ytr = Ytr + curr_y

    Ytr = np.array(Ytr)
    print(Xtr.shape)
    print(Ytr.shape)

    dict = unpickle(path + 'test_batch')
    Xte = dict[b'data']
    Yte = np.array(dict[b'labels'])

    print(Xte.shape)
    print(Yte.shape)

    return Xtr, Ytr, Xte, Yte
