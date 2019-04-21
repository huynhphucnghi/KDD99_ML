import numpy as np
import matplotlib.pyplot as plt

from readcsv import readfileCSV
from utils import *


def main():
    # Non crossvalidation
    print('Non-crossvalidation:')
    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        dataset, target, test_size=0.2, random_state=0)
    X_train = np.asarray(X_train).astype(np.float)
    X_test = np.asarray(X_test).astype(np.float)
    y_train = np.asarray(y_train).astype(np.float)
    y_test = np.asarray(y_test).astype(np.float)
    #

    # Model
    # model = NBmodel()
    model = KNNmodel()

    # fit model
    model = model.fit(X_train, y_train)
    # Model score
    score = model.score(X_test, y_test)
    print('test score: ', score)

    y_pred = model.predict(X_test)
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

    # Crossvalidation
    print('Crossvalidation:')
    crossvalidation(model, dataset, target)


if __name__ == "__main__":
    dataset, target = readfileCSV()
    main()
