import numpy as np
import time
import pandas as pd


def readTrainCSV(inputcsv, fea_sel=0):
    # preselection for running
    df = pd.read_csv(inputcsv,
                     #  nrows=100000,
                     #  usecols=range(len(full_categories))
                     )
    # Converter

    # Frequency per label
    print('Frequency per label:')
    unique, count = np.unique(df[df.columns[-1]], return_counts=True)
    print(np.asarray((unique, count)))

    for i in range(len(df.columns)):
        df[df.columns[i]] = df[df.columns[i]].factorize()[0]

    # Devide
    dataset = df.iloc[:, 1:-1]
    target = df.iloc[:, -1]
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


def loadModel():

    # Model
    model = NBmodel()
    # model = KNNmodel()

    # fit model
    # model = model.fit(X_train, y_train)

    # from sklearn.externals import joblib
    # filename = './model/model1.sav'
    # joblib.dump(model, filename)

    return model


def crossvalidation(model, dataset, target):
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, dataset, target, cv=5)
    print(scores)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
