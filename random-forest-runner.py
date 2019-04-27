#%%
import numpy as np
from readcsv import readfileCSV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import timeit
# data, target = readfileCSV()

# print(type(data))
# print(np.unique(target))

MAX_SIZE = 488736
SIZE = MAX_SIZE

dataset, target = readfileCSV()
dataset = dataset[:SIZE, :]
target = target[:SIZE]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    dataset, target, test_size=0.2, random_state=0)
X_train = np.asarray(X_train).astype(np.float)
X_test = np.asarray(X_test).astype(np.float)
y_train = np.asarray(y_train).astype(np.float)
y_test = np.asarray(y_test).astype(np.float)

print("start train")

parameters = {'n_estimators': [70,80,90], 'max_depth': [5,7], 'min_samples_split':[3,4]}
rf = RandomForestClassifier(n_estimators = 100, max_depth = 10, min_samples_split = 2, max_features = None, oob_score = True, random_state = 123456)
time_begin = timeit.default_timer()
gs_rf = GridSearchCV(rf, parameters, n_jobs=-1)
gs_rf = gs_rf.fit(X_train, y_train)
runtime = timeit.default_timer() - time_begin
rf = gs_rf.best_estimator_
y_pred = rf.predict(X_test)
score = accuracy_score(y_test, y_pred)

print("Accuracy: ", score * 100)
print("Runtime: ", runtime)


#%%
print(rf)
time_begin = timeit.default_timer()
y_pred = rf.predict(X_test)
runtime = timeit.default_timer() - time_begin
#%%

report = metrics.classification_report(y_pred, y_test)

print(report)
print(runtime)