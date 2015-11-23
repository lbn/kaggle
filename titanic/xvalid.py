from config import config

import numpy as np
from sklearn import metrics
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class Report(dict):
    def __init__(self):
        super(Report, self).__init__()

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

class CrossValidator(object):
    def __init__(self, data):
        self.data = data

    def run(self, clf):
        X, y = clf.preprocess(self.data)
        skf_config = config.xvalidation
        self.skf = StratifiedShuffleSplit(y, **skf_config)
        self.X = X
        self.y = y

        report = Report()

        X, y = self.X, self.y

        y_true = []
        y_predicted = []

        for train_index, test_index in self.skf:
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

            y_true += list(y_test)

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            y_predicted += list(y_pred)

        report.summary = classification_report(y_true, y_predicted, target_names=("dead", "alive"))
        cm = confusion_matrix(y_true, y_predicted)
        report.confusion_matrix = cm
        report.confusion_matrix_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        report.accuracy = accuracy_score(y_true, y_predicted)

        return report
