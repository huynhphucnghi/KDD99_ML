import numpy as np
import time
import pandas as pd
import pickle
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=1)
np.set_printoptions(suppress=True)

SIZE = 20000


def preprocess(array):
    dataset = array[:, 1:-1]
    target = array[:, -1]
    return dataset, target


def readTrainCSV(inputcsv, fea_sel=0):
    # preselection for running
    df = pd.read_csv(inputcsv,
                     nrows=SIZE,
                     #   usecols=range(len(full_categories))
                     )
    # Converter

    # Frequency per label
    print('Frequency per label:')
    unique, count = np.unique(df[df.columns[-1]], return_counts=True)
    print(np.asarray((unique, count)))

    # Devide
    data = np.asarray(df)
    # dataset = df.iloc[:, 1:-1]
    # target = df.iloc[:, -1]

    # Preprocessing
    dataset, target = preprocess(data)

    # Label encoder
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    target = le.fit_transform(target)
    print('Labels dictionary:')
    labels_dict = dict(zip(range(len(le.classes_)), list(le.classes_)))
    print(labels_dict)

    print('Shape of dataset: ', dataset.shape)

    if fea_sel:
        # Feature selection
        num_feature = 40
        print('Feature Selection (top', num_feature, 'feature):')
        from sklearn.datasets import load_digits
        from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, f_classif, mutual_info_classif

        # Use SelectKBest
        selector = SelectKBest(mutual_info_classif, k=num_feature)
        # Or Use SelectPercentile
        # selector = SelectPercentile(chi2, percentile=50)

        fit_selector = selector.fit(dataset, target)
        # Score of selection
        dfscores = pd.DataFrame(fit_selector.scores_)
        dfcolumns = pd.DataFrame(dataset.columns)
        featureScores = pd.concat([dfcolumns, dfscores], axis=1)
        # naming the dataframe columns
        featureScores.columns = ['Specs', 'Score']
        # print 10 best features
        print(featureScores.nlargest(num_feature, 'Score'))

        index = selector.get_support(indices=True)
        print(index)
        # Selection Transform
        dataset = selector.transform(dataset)

    print(dataset.shape)

    return dataset, target


def load_model(model_file):
    return pickle.load(open(model_file, 'rb'))


def custom_predict(model_file, X_test):
    model = load_model(model_file)
    return model.predict(X_test)
