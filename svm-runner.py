import numpy as np
from readcsv import readfileCSV
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import metrics
# data, target = readfileCSV()

# print(type(data))
# print(np.unique(target))

SIZE = 50000

def main():
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        dataset, target, test_size=0.3, random_state=0)
    X_train = np.asarray(X_train).astype(np.float)
    X_test = np.asarray(X_test).astype(np.float)
    y_train = np.asarray(y_train).astype(np.float)
    y_test = np.asarray(y_test).astype(np.float)

    print("start train")
    #SVM
    gs_clf = SVC()
    parameters = {'kernel':('linear', 'rbf'), 'C':[1,2]}
    gs_clf = GridSearchCV(gs_clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(X_train, y_train)

    best_clf = gs_clf.best_estimator_
    print(best_clf)
    # scores = cross_val_score(clf, X_test, y_test, cv = 5)
    pred = best_clf.predict(X_test)
    score = accuracy_score(y_test, pred)

    print("finish train")
    report = metrics.classification_report(y_test, pred)
    print(report)
    print('Scores: ', score * 100)


if __name__ == "__main__":
    dataset, target = readfileCSV()
    dataset = dataset[:SIZE, :]
    target = target[:SIZE]
    main()

#In[]
# from sklearn import metrics

# report = metrics.