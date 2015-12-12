import os
from collections import Counter
from sklearn import preprocessing
from sklearn.linear_model import SGDRegressor

import pandas as pd
import numpy as np

def extract_titles(dataset):
    return [name.split(', ')[1].split('.')[0] for name in dataset["Name"]]

def common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


data_dir = os.path.dirname(__file__)
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))

# Add enriched fields
train["Title"] = extract_titles(train)
test["Title"] = extract_titles(test)

# Encoders for category features
sex_enc = preprocessing.LabelBinarizer()
sex_enc.fit(train["Sex"])

title_enc = preprocessing.LabelBinarizer()
title_enc.fit(train["Title"])

train["Embarked"].fillna(common(train["Embarked"]), inplace=True)
# Impute test dataset N/A values using the most common values in train
test["Embarked"].fillna(common(train["Embarked"]), inplace=True)

embarked_enc = preprocessing.LabelBinarizer()
embarked_enc.fit(train["Embarked"])

train["Pclass"].fillna(common(train["Pclass"]), inplace=True)
# Impute test dataset N/A values using the most common values in train
test["Pclass"].fillna(common(train["Pclass"]), inplace=True)

pclass_enc = preprocessing.LabelBinarizer()
pclass_enc.fit(train["Pclass"])

parch_enc = preprocessing.LabelBinarizer()
parch_enc.fit(train["Parch"])


def predict_age():
    mask = ~np.isnan(train["Age"])
    age_train = train[mask]
    age_test = train[~mask]

    features = []
    features.append(embarked_enc.transform(age_train["Embarked"]))
    features.append(sex_enc.transform(age_train["Sex"]))
    features.append(title_enc.transform(age_train["Title"]))
    features.append(pclass_enc.transform(age_train["Pclass"]))

    age_clf = SGDRegressor()
    X = np.hstack(features)
    y = np.array(train["Age"][mask]).T
    age_clf.fit(X, y)

    features = []
    features.append(embarked_enc.transform(age_test["Embarked"]))
    features.append(sex_enc.transform(age_test["Sex"]))
    features.append(title_enc.transform(age_test["Title"]))
    features.append(pclass_enc.transform(age_test["Pclass"]))

    ages = age_clf.predict(np.hstack(features))
    j = 0
    for i in range(len(train)):
        if ~mask[i]:
            train.loc[i, "Age"] = ages[j]
            j += 1


# predict_age()
train["Age"].fillna(np.mean(train["Age"]), inplace=True)
# Impute test dataset N/A values using the mean values in train
test["Age"].fillna(np.mean(train["Age"]), inplace=True)

def extract_features(dataset):
    embarked = embarked_enc.transform(dataset["Embarked"])
    sex = sex_enc.transform(dataset["Sex"])
    title = title_enc.transform(dataset["Title"])
    pclass = pclass_enc.transform(dataset["Pclass"])

    sibsp = np.array((dataset["SibSp"],)).T
    parch = np.array((dataset["Parch"],)).T
    # parch = np.array(dataset["Parch"] > 0, dtype=np.uint8).reshape(-1, 1)

    age = np.array((dataset["Age"],)).T
    fare = np.array((dataset["Fare"],)).T

    features = [embarked, sex, pclass, title]
    return np.hstack(features)

def split(dataset):
    X = extract_features(dataset)
    y = dataset["Survived"]
    return X, np.array(y)

def categorise(vec):
    choices = tuple(Counter(vec).keys())
    return np.vstack([vec == choice for choice in choices]).T
