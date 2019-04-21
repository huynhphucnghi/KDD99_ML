import csv
import json
import numpy as np
from pprint import pprint
import pandas as pd

from utils import *

path = 'Data/'
# filecsv = 'data.csv'
filecsv = 'kddcup.data.corrected.csv'
class_list = ['smurf.', 'land.', 'pod.',
              'teardrop.', 'neptune.', 'back.', 'normal.']


def readfileCSV():

    dictin = dict((value, index) for index, value in enumerate(categories))
    # df = pd.read_csv(path + filecsv, names=categories, usecols=range(len(categories)))
    df = pd.read_csv(path + filecsv, names=categories,
                     nrows=100000, usecols=range(len(categories)))
    df = df[[(x in class_list) for x in df['label']]]

    # Converter
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    data = df[df.columns[:]].apply(le.fit_transform)

    # Devide
    dataset = data.iloc[:, :-1]
    target = data.iloc[:, -1]
    print('Shape of dataset: ', dataset.shape)

    # Frequency per label
    print('Frequency per label:')
    unique, count = np.unique(df['label'], return_counts=True)
    print(np.asarray((unique, count)))

    # Feature selection
    num_feature = 10
    print('Feature Selection (top', num_feature, 'feature):')

    from sklearn.datasets import load_digits
    from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2

    # Use SelectKBest
    selector = SelectKBest(chi2, k=num_feature)
    # Or Use SelectPercentile
    # selector = SelectPercentile(chi2, percentile=50)

    fit_selector = selector.fit(dataset, target)
    # Score of selection
    dfscores = pd.DataFrame(fit_selector.scores_)
    dfcolumns = pd.DataFrame(dataset.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    # print 10 best features
    print(featureScores.nlargest(num_feature, 'Score'))

    # Selection Transform
    selected_dataset = selector.transform(dataset)

    index = selector.get_support(indices=True)
    print(index)
    print(selected_dataset.shape)

    return selected_dataset, target
