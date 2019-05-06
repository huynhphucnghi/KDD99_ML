import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import *


filecsv = 'Data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'


def dataSplit(filecsv):

    dataset, target = readTrainCSV(filecsv, fea_sel=0)
    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        dataset, target, test_size=0.2, random_state=0)
    X_train = np.asarray(X_train).astype(np.float)
    X_test = np.asarray(X_test).astype(np.float)
    y_train = np.asarray(y_train).astype(np.float)
    y_test = np.asarray(y_test).astype(np.float)

    return X_train, y_train, X_test, y_test


def train(model, filename, **kwargs):
    X_train, y_train, X_test, y_test = dataSplit(filename)
    if model == 'KNN':
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(**kwargs)
    if model == 'NB':
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()

    # fit model
    model = model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print('test score: ', score)

    start = time.time()
    y_pred = model.predict(X_test)
    end = time.time()
    print('Total classification time: ', end - start)
    # Number of missing
    print("Number of mislabeled points out of a total %d points : %d"
          % (X_test.shape[0], (y_test != y_pred).sum()))
    # Metric
    print('Precision, Recall, Fscore, Support per class:')
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, fscore, support = precision_recall_fscore_support(
        y_test, y_pred)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print('Precision:   {}'.format(precision))
    print('Recall:      {}'.format(recall))
    print('Fscore:      {}'.format(fscore))
    np.set_printoptions(formatter={'int': '{: 6d}'.format})
    print('Support:     {}'.format(support))
    # # Frequency per label
    # print('Frequency per label:')
    # unique, count = np.unique(y_pred, return_counts=True)
    # print(np.asarray(([label[int(item)] for item in unique], count)))
    # counter = 0
    # for pred in y_pred:
    #     if pred == 2.0:
    #         # print(label[int(pred)], end=' ')
    #         print(counter)
    #     counter += 1
    #     pass
    # print()


if __name__ == "__main__":
    model_name = 'KNN'
    train(model_name, filecsv, n_neighbors=1)
