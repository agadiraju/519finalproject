import math
import numpy as np
import sys

from import_train import import_training_file
from import_train import rmsle
from sklearn.naive_bayes import MultinomialNB


def simple_naive_bayes(X, y):
  n, _ = X.shape
  nTrain = int(0.5*n)  #training on 50% of the data
  Xtrain = X[:nTrain,:]
  ytrain = y[:nTrain]
  Xtest = X[nTrain:,:]
  ytest = y[nTrain:]

  clf = MultinomialNB().fit(Xtrain, ytrain)
  predict_y = clf.predict(Xtest)
  print ytest
  print predict_y
  print rmsle(ytest, predict_y)

if __name__ == '__main__':
  (X, y) = import_training_file(sys.argv[1], True)
  simple_naive_bayes(X, y)