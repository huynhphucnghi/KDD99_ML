from sklearn.model_selection import GridSearchCV
import pickle
import numpy as np
import time

class Trainer():
    def __init__(self, model_name = 'KNN',  **hyper_params):
        self.model_name = model_name
        self.params = hyper_params
        self.is_gridsearch = 0
        if model_name == 'KNN':
            from sklearn.neighbors import KNeighborsClassifier
            self.model = KNeighborsClassifier(**hyper_params)
        if model_name == 'NB':
            from sklearn.naive_bayes import GaussianNB
            self.model = GaussianNB(**hyper_params)
        if model_name == 'SVM':
            from sklearn.svm import SVC
            self.model = SVC(**hyper_params)
        
    def gridsearchCV(self, params):
        self.model = GridSearchCV(self.model, params , n_jobs=-1)
        self.is_gridsearch = 1
    
    def fit(self, X, y):
        self.model.fit(X, y)
        if(self.is_gridsearch == 1):
            self.model = self.model.best_estimator_
        

    def predict(self, X):
        begin_time = time.time()
        self.y_pred = self.model.predict(X)
        self.pred_time = time.time() - begin_time

    def save_model(self, trained_file_name):
        pickle.dump(self.model, open(trained_file_name, 'wb'))

    def load_model(self, trained_file_name):
        self.model = pickle.load(open(trained_file_name, 'rb'))


    def report(self, target):
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, fscore, support = precision_recall_fscore_support(self.y_pred, target)
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print('Precision:   {}'.format(precision))
        print('Recall:      {}'.format(recall))
        print('Fscore:      {}'.format(fscore))
        np.set_printoptions(formatter={'int': '{: 6d}'.format})
        print('Support:     {}'.format(support))
