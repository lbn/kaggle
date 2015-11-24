from collections import Counter

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from config import config

def features_1(dataset):
    mask = np.isfinite(dataset["Age"])
    mask &= [type(p) == str for p in dataset["Embarked"]]
    mask &= np.isfinite(dataset["Survived"])

    dataset = dataset[mask]

    embarked = categorise(dataset["Embarked"])
    sex = categorise(dataset["Sex"])
    pclass = categorise(dataset["Pclass"])
    sibsp = np.array((dataset["SibSp"],)).T
    parch = np.array((dataset["Parch"],)).T
    age = np.array((dataset["Age"],)).T

    return np.hstack([embarked, sex, pclass, sibsp, parch, age]), np.array(dataset["Survived"])

def categorise(vec):
    choices = tuple(Counter(vec).keys())
    return np.vstack([vec == choice for choice in choices]).T

class BasicSVM(object):
    def __init__(self):
        self.clf = LinearSVC()

    def preprocess(self, dataset):
        return features_1(dataset)

    def fit(self, X, y):
        return self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

class NB(object):
    def __init__(self):
        self.clf = GaussianNB()

    def preprocess(self, dataset):
        return features_1(dataset)

    def fit(self, X, y):
        return self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

class DTC(object):
    def __init__(self):
        self.clf = DecisionTreeClassifier()

    def preprocess(self, dataset):
        return features_1(dataset)

    def fit(self, X, y):
        return self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

class AdaBoost(object):
    def __init__(self):
        self.clf = AdaBoostClassifier()

    def preprocess(self, dataset):
        return features_1(dataset)

    def fit(self, X, y):
        return self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

class GradientBoosting(object):
    def __init__(self):
        self.clf = GradientBoostingClassifier()

    def preprocess(self, dataset):
        return features_1(dataset)

    def fit(self, X, y):
        return self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

class RandomForest(object):
    def __init__(self):
        self.clf = RandomForestClassifier()

    def preprocess(self, dataset):
        return features_1(dataset)

    def fit(self, X, y):
        return self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)
