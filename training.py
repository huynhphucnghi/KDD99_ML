import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from model import Trainer



filecsv = 'Data/Wednesday-workingHours.pcap_ISCX.csv'

if __name__ == "__main__":

    dataset, target = readTrainCSV(filecsv, fea_sel=0)

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        dataset, target, test_size=0.2, random_state=0)
    X_train = np.asarray(X_train).astype(np.float)
    X_test = np.asarray(X_test).astype(np.float)
    y_train = np.asarray(y_train).astype(np.float)
    y_test = np.asarray(y_test).astype(np.float)



    model_name = 'SVM'

    classifer = Trainer(model_name, kernel='rbf', C=1, gamma='scale')

    # classifer.load_model('kNN_classifier.txt')


    # parameters = {'kernel':('linear', 'rbf'), 'C':[1,2]}

    # classifer.gridsearchCV(parameters)

    classifer.fit(X_train, y_train)
    classifer.predict(X_test)
    classifer.report(y_test)
    print(classifer.model)
    classifer.save_model('SVM_classifier.sav')

