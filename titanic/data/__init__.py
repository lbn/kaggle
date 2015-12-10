import os
from collections import Counter

import pandas

import numpy as np


data_dir = os.path.dirname(__file__)
train = pandas.read_csv(os.path.join(data_dir, "train.csv"))
test = pandas.read_csv(os.path.join(data_dir, "test.csv"))

def common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def extract_features(dataset, exclude=True):
    mask = None
    if exclude:
        mask = np.isfinite(dataset["Age"])
        mask &= [type(p) == str for p in dataset["Embarked"]]
        if "Survived" in dataset:
            mask &= np.isfinite(dataset["Survived"])

        dataset = dataset[mask]
    else:
        print("Common embarked")
        print(common(dataset["Embarked"]))
        print("Common Pclass")
        print(common(dataset["Pclass"]))

        dataset["Age"].fillna(np.mean(dataset["Age"]), inplace=True)
        dataset["Embarked"].fillna(common(dataset["Embarked"]), inplace=True)
        dataset["Sex"].fillna(common(dataset["Sex"]), inplace=True)
        dataset["Pclass"].fillna(common(dataset["Pclass"]), inplace=True)
        dataset["SibSp"].fillna(np.mean(dataset["SibSp"]), inplace=True)
        dataset["Parch"].fillna(np.mean(dataset["Parch"]), inplace=True)
        dataset["Fare"].fillna(np.mean(dataset["Fare"]), inplace=True)

    embarked = categorise(dataset["Embarked"])
    sex = categorise(dataset["Sex"])
    pclass = categorise(dataset["Pclass"])
    sibsp = np.array((dataset["SibSp"],)).T
    parch = np.array((dataset["Parch"],)).T
    age = np.array((dataset["Age"],)).T
    fare = np.array((dataset["Fare"],)).T

    # return np.hstack([embarked, sex, pclass, sibsp, parch, age, fare]), mask
    return np.hstack([sex, pclass, fare]), mask

def split(dataset):
    X, mask = extract_features(dataset, exclude=False)
    y = dataset["Survived"]
    if mask is not None:
        y = y[mask]
    return X, np.array(y)

def categorise(vec):
    choices = tuple(Counter(vec).keys())
    return np.vstack([vec == choice for choice in choices]).T
