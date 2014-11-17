import math
import numpy as np
import sys

from import_train import import_training_file
from import_train import rmsle
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier

def decision_tree(X, y):
  n, _ = X.shape
  nTrain = int(0.5*n)  #training on 50% of the data
  Xtrain = X[:nTrain,:]
  ytrain = y[:nTrain]
  Xtest = X[nTrain:,:]
  ytest = y[nTrain:]

  clf_1 = DecisionTreeRegressor(max_depth=None)
  clf_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=None),
                          n_estimators=500)
  clf_4 = RandomForestClassifier(n_estimators=10, max_depth=None,
                          min_samples_split=1, random_state=0)
  clf_5 = ExtraTreesClassifier(n_estimators=10, max_depth=None,
                          min_samples_split=1, random_state=0)
  clf_3 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                          max_depth=1, random_state=0)

  print "finished generating tree"

  clf_1.fit(Xtrain, ytrain)
  clf_2.fit(Xtrain, ytrain)
  clf_3.fit(Xtrain, ytrain)
  clf_4.fit(Xtrain, ytrain)
  clf_5.fit(Xtrain, ytrain)


  print 'Finished fitting'


  y1 = clf_1.predict(Xtest)
  y2 = clf_2.predict(Xtest)
  y3 = clf_3.predict(Xtest)
  y4 = clf_4.predict(Xtest)
  y5 = clf_5.predict(Xtest)

  print "regular decision tree"
  print rmsle(ytest, y1)
  print "boosted decision tree"
  print rmsle(ytest, y2)
  print "gradient tree boosting"
  print rmsle(ytest, y3)
  print "random forest classifier"
  print rmsle(ytest, y4)
  print "extra trees classifier"
  print rmsle(ytest, y5)


if __name__ == '__main__':
  (X, y) = import_training_file(sys.argv[1], True)
  decision_tree(X, y)
