import math
import numpy as np
import sys

from import_train import import_training_file
from import_train import rmsle
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor


def decision_tree(X, y):
  n, _ = X.shape
  nTrain = int(0.5*n)  #training on 50% of the data
  Xtrain = X[:nTrain,:]
  ytrain = y[:nTrain]
  Xtest = X[nTrain:,:]
  ytest = y[nTrain:]

  clf_1 = DecisionTreeRegressor(max_depth=4)
  clf_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300)
  clf_1.fit(Xtrain, ytrain)
  clf_2.fit(Xtrain, ytrain)

  y1 = clf_1.predict(Xtest)
  y2 = clf_2.predict(Xtest)

  print "regular decision tree"
  print rmsle(ytest, y1)
  print "boosted decision tree"
  print rmsle(ytest, y2)


if __name__ == '__main__':
  (X, y) = import_training_file(sys.argv[1], True)
  decision_tree(X, y)
