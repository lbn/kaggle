import time
import datetime

import numpy as np
import pandas as pd

from xvalid import CrossValidator
import data
from classifier import BasicSVM, NB, DTC, AdaBoost, GradientBoosting, RandomForest, XGB


def xvalidate():
    xv = CrossValidator(data.train)

    start = time.time()

    classifiers = (GradientBoosting, RandomForest, XGB)
    for Classifier in classifiers:
        report = xv.run(Classifier)
        print(report)

    time_taken = time.time() - start

    print("-"*80)
    print("Completed in {:.4f}s on {}".format(time_taken, str(datetime.datetime.now())))

def test():
    clf = GradientBoosting
    X_train, y_train = data.split(data.train)
    X_test = data.extract_features(data.test)

    clf.fit(X_train, y_train)
    labels = clf.predict(X_test)
    pd.DataFrame({
        "PassengerId": np.array(data.test["PassengerId"]),
        "Survived": labels
    }).to_csv("submit.csv", index=False)

def main():
    # xvalidate()
    test()

if __name__ == "__main__":
    main()
