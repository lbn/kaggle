import os
from collections import Counter

import numpy as np
import pandas


data_dir = os.path.dirname(__file__)
train = pandas.read_csv(os.path.join(data_dir, "train.csv"))
test = pandas.read_csv(os.path.join(data_dir, "test.csv"))

def split(dataset):
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
