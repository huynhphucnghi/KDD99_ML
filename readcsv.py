import csv
import json
import numpy as np 
from pprint import pprint
import pandas as pd

from utils import *

path = 'Data/'
filecsv = 'data.csv'
class_list = ['smurf.', 'land.', 'pod.', 'teardrop.', 'neptune.', 'back.', 'normal.']

def readfileCSV():

    dictin = dict((value, index) for index, value in enumerate(categories))
    df = pd.read_csv(path + filecsv, names=categories, usecols=range(len(categories)))
    # df = pd.read_csv(path + filecsv, names=categories, nrows= 100000, usecols=range(len(categories)))
    df = df[[(x in class_list) for x in df['label']]]

    # Converter
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    data = df[df.columns[:]].apply(le.fit_transform)

    # Devide
    target = data.iloc[:,-1].values
    dataset = data.iloc[:,:-1].values
    print(dataset.shape)

    # Frequency per label
    unique, count = np.unique(target, return_counts=True)
    print(np.asarray((unique, count)))

    # Future selection
    from sklearn.datasets import load_digits
    from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
    # Use SelectKBest
    dataset = SelectKBest(chi2, k=20).fit_transform(dataset, target)
    # Use SelectPercentile
    # dataset = SelectPercentile(chi2, percentile=50).fit_transform(dataset, target)

    print(dataset.shape)

    return dataset, target

    