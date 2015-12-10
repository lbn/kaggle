from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from xgb import XGBoostClassifier


BasicSVM = Pipeline([("SVM (linear)", LinearSVC())])
NB = Pipeline([("Gaussian NB Bayes", GaussianNB())])
SGD = Pipeline([("Stochastic Gradient Descent", SGDClassifier())])
DTC = Pipeline([("Decision Tree", DecisionTreeClassifier())])
AdaBoost = Pipeline([("Ada Boost", AdaBoostClassifier())])
GradientBoosting = Pipeline([("Gradient Boosting", GradientBoostingClassifier())])
XGB = Pipeline([("XGBoost", XGBoostClassifier(num_class=2, silent=1))])
RandomForest = Pipeline([("Random Forest", RandomForestClassifier())])
