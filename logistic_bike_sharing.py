import math
import numpy as np
import sys

from import_train import import_training_file
from import_train import rmsle
from sklearn import linear_model


def logistic_regression(X, y):
  n, _ = X.shape
  nTrain = int(0.5*n)  #training on 50% of the data
  Xtrain = X[:nTrain,:]
  ytrain = y[:nTrain]
  Xtest = X[nTrain:,:]
  ytest = y[nTrain:]

  logreg = linear_model.LogisticRegression(C=1e5)
  logreg.fit(Xtrain, ytrain)

  y1 = logreg.predict(Xtest)

  print rmsle(ytest, y1)


if __name__ == '__main__':
  (X, y) = import_training_file(sys.argv[1], True)
  logistic_regression(X, y)
