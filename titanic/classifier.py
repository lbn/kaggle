from collections import Counter

import numpy as np
from sklearn.svm import LinearSVC

from config import config

def categorise(vec):
    choices = tuple(Counter(vec).keys())
    return np.vstack([vec == choice for choice in choices]).T

class BasicSVM(object):
    def __init__(self):
        self.clf = LinearSVC()

    def preprocess(self, dataset):
        mask = np.isfinite(dataset["Age"])
        mask &= [type(p) == str for p in dataset["Embarked"]]
        mask &= np.isfinite(dataset["Survived"])

        dataset = dataset[mask]

        embarked = categorise(dataset["Embarked"])
        sex = categorise(dataset["Sex"])
        pclass = categorise(dataset["Pclass"])
        sibsp = np.array((dataset["SibSp"],)).T
        parch = np.array((dataset["Parch"],)).T

        return np.hstack([embarked, sex, pclass, sibsp, parch]), np.array(dataset["Survived"])

    def fit(self, X, y):
        return self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)
