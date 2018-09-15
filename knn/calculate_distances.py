

def calculate_distance(Xtr, X, degree):
    import numpy as np
    if degree == 1:
        distances = np.sum(np.abs(Xtr - X), axis=1)
    else:
        distances = np.power(np.sum(np.power(Xtr - X, degree), axis=1), 1 / degree)

    return distances


def compute_big_matrix(Xtr_rows, Xte_rows):
    import time
    num_test = Xte_rows.shape[0]
    big_matrix = []
    for i in range(num_test):
        print(i)
        print(time.time())
        dists = calculate_distance(Xtr_rows, Xte_rows[i, :], degree=2)
        big_matrix.append(dists)

    return big_matrix


def save(obj, file):
    import pickle
    import sys
    max_bytes = 2 ** 31 - 1
    bytes_out = pickle.dumps(obj, protocol=4)
    n_bytes = sys.getsizeof(bytes_out)
    with open(file, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def load(file):
    import pickle
    import os
    max_bytes = 2 ** 31 - 1
    try:
        input_size = os.path.getsize(file)
        bytes_in = bytearray(0)
        with open(file, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        obj = pickle.loads(bytes_in)
    except:
        return None
    return obj
