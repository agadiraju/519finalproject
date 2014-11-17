import math
import numpy as np
import sys

from import_train import import_training_file
from import_train import rmsle
from sklearn.linear_model import LogisticRegression


def logistic_regression(X, y):
  n, _ = X.shape
  nTrain = int(0.5*n)  #training on 50% of the data
  Xtrain = X[:nTrain,:]
  ytrain = y[:nTrain]
  Xtest = X[nTrain:,:]
  ytest = y[nTrain:]

  for i, C in enumerate(10. ** np.arange(1, 6)):
    clf_l1_LR = LogisticRegression(C=C, penalty='l1', tol=0.01)
    clf_l2_LR = LogisticRegression(C=C, penalty='l2', tol=0.01)
    clf_l1_LR.fit(Xtrain, ytrain)
    clf_l2_LR.fit(Xtrain, ytrain)


    y1 = clf_l1_LR.predict(Xtest)
    y2 = clf_l2_LR.predict(Xtest)

    #L1 penalty
    print "L1 Penalty with C=" + str(C)
    print rmsle(ytest, y1)
    print "L2 Penalty with C=" + str(C)
    #L2 penalty
    print rmsle(ytest, y2)

# logreg = LogisticRegression(C=1e5)
# logreg.fit(Xtrain, ytrain)
# y3 = logreg.predict(Xtest)
# print "no penalty, C=" + str(1e5)
# print rmsle(ytest,y3)



if __name__ == '__main__':
  (X, y) = import_training_file(sys.argv[1], True)
  logistic_regression(X, y)
