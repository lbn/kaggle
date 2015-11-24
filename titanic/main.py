import time
import datetime

from xvalid import CrossValidator
import data
from classifier import BasicSVM, NB, DTC, AdaBoost, GradientBoosting, RandomForest


def main():
    xv = CrossValidator(data.train)

    start = time.time()

    classifiers = (BasicSVM, NB, DTC, AdaBoost, GradientBoosting, RandomForest)
    for Classifier in classifiers:
        report = xv.run(Classifier())
        print(report)

    time_taken = time.time() - start

    print("-"*80)
    print("Completed in {:.4f}s on {}".format(time_taken, str(datetime.datetime.now())))


if __name__ == "__main__":
    main()
