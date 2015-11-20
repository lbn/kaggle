from collections import Counter

import numpy as np
from sklearn.svm import LinearSVC

import data
from config import config
from xvalid import CrossValidator

def categorise(vec):
    choices = tuple(Counter(vec).keys())
    return np.vstack([vec == choice for choice in choices]).T

def get_features(dataset):
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


def main():
    X, y = get_features(data.train)
    xv = CrossValidator(X, y)
    clf = LinearSVC()
    report = xv.run(clf)

    print("# Summary")
    print("\n```")
    print(report.summary)
    print("```\n")

    print("# Accuracy")
    print("\n```")
    print(report.accuracy)
    print("```\n")

if __name__ == "__main__":
    main()
