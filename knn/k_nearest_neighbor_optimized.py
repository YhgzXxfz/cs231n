import sys

import numpy as np


class KNearestNeighbor:
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix
        pass

    def train(self, Xtr, Ytr):
        self.Xtr = Xtr
        self.Ytr = Ytr

    def predict(self, X, degree=1, k=1):
        num_test = X.shape[0]

        Ypred = np.zeros(num_test, dtype=self.Ytr.dtype)
        for i in range(num_test):
            # print(i)
            # print(time.time())
            distances = self.distance_matrix[i]
            # print(time.time())
            max_dist = max(distances)
            # print(time.time())
            indexes = []
            for j in range(k):
                min_index = np.argmin(distances)
                # print(time.time())
                indexes.append((min_index, distances[min_index], self.Ytr[min_index]))
                distances[min_index] = max_dist

            # print(time.time())
            voted_index = self.vote_index(indexes)
            Ypred[i] = self.Ytr[voted_index]

        return Ypred

    def vote_index(self, indexes):
        label_occur = {}
        label_index = {}
        label_dist = {}
        occur_label = {}

        max_occur = 0
        for i, dist, label in indexes:
            if label not in label_occur:
                label_occur[label] = 0
                label_index[label] = []
                label_dist[label] = []

            label_index[label].append(i)
            label_dist[label].append(dist)
            label_occur[label] = label_occur[label] + 1

            if label_occur[label] > max_occur:
                max_occur = label_occur[label]

        for (label, occur) in label_occur.items():
            if occur not in occur_label:
                occur_label[occur] = []
            occur_label[occur].append(label)

        candidates = occur_label[max_occur]
        if len(candidates) == 1:
            return label_index[candidates[0]][0]
        else:
            min_mean_dist = sys.maxsize
            result = 0
            for cand in candidates:
                mean_dist = np.mean(label_dist[cand])
                if mean_dist < min_mean_dist:
                    result = cand
                    min_mean_dist = mean_dist

            return label_index[result][0]
