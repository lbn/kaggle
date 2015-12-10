from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline


BasicSVM = Pipeline([('SVM (linear)', LinearSVC())])
NB = Pipeline([('Gaussian NB Bayes', GaussianNB())])
DTC = Pipeline([('Decision Tree', DecisionTreeClassifier())])
AdaBoost = Pipeline([('Ada Boost', AdaBoostClassifier())])
GradientBoosting = Pipeline([('Gradient Boosting', GradientBoostingClassifier())])
RandomForest = Pipeline([('Random Forest', RandomForestClassifier())])
