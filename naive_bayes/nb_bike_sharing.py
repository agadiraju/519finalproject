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

def harder_naive_bayes(X, y):
  # divide into atemp classes
  X = np.delete(X, 4, axis = 1)  # remove temp column (not independent from atemp)
  
  for row in X:
    # reclassify atemp
    if row[4] >= 40.0:
      row[4] = 4
    elif row[4] >= 30.0:
      row[4] = 3
    elif row[4] >= 20.0:
      row[4] = 2
    elif row[4] >= 10.0:
      row[4] = 1
    else:
      row[4] = 0

    # reclassify humidity
    # print row[5]
    # if row[5] == 0:
    #   row[5] == .00001
    # row[5] = np.floor(100 / row[5])
    if row[5] >= 80:
      row[5] = 4
    elif row[5] >= 60:
      row[5] = 3
    elif row[5] >= 40:
      row[5] = 2
    elif row[5] >= 20:
      row[5] = 1
    else:
      row[5] = 0

    # reclassify wind speed
    if row[6] >= 30.0:
      row[6] = 3
    elif row[6] >= 20.0:
      row[6] = 2
    elif row[6] >= 10.0:
      row[6] = 1
    else:
      row[6] = 0

  print X[0]
  print y[0]
  return simple_naive_bayes(X, y)

if __name__ == '__main__':
  (X, y) = import_training_file(sys.argv[1], True)
  simple_naive_bayes(X, y)
  print '------------------'
  harder_naive_bayes(X, y)