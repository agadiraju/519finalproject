import math
import numpy as np
import sys

from import_train import import_training_file
from import_train import rmsle
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor

def decision_tree(X, y1, y2, y3):
  n, _ = X.shape
  nTrain = int(0.5*n)  #training on 50% of the data
  Xtrain = X[:nTrain,:]
  ytrain = y1[:nTrain]
  ytrain_registered = y2[:nTrain]
  ytest_registered = y2[nTrain:]
  ytrain_casual = y3[:nTrain]
  ytest_casual = y3[nTrain:]
  Xtest = X[nTrain:,:]
  ytest = y1[nTrain:]

  #regular

  clf_1 = DecisionTreeRegressor(max_depth=None)
  clf_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=None),
                          n_estimators=500)
  clf_4 = RandomForestRegressor(n_estimators=500, max_depth=None,
                          min_samples_split=1, random_state=0)
  clf_5 = ExtraTreesRegressor(n_estimators=500, max_depth=None,
                          min_samples_split=1, random_state=0)
  clf_3 = GradientBoostingRegressor(n_estimators=500,
                          max_depth=None, random_state=0)

  print "finished generating tree"

  clf_1.fit(Xtrain, ytrain_registered)
  clf_2.fit(Xtrain, ytrain_registered)
  clf_3.fit(Xtrain, ytrain_registered)
  clf_4.fit(Xtrain, ytrain_registered)
  clf_5.fit(Xtrain, ytrain_registered)


  print 'Finished fitting'


  dt_regular = clf_1.predict(Xtest)
  ada_regular = clf_2.predict(Xtest)
  grad_regular = clf_3.predict(Xtest)
  rf_regular = clf_4.predict(Xtest)
  et_regular = clf_5.predict(Xtest)

  #casual
  print "finished generating tree"

  clf_1.fit(Xtrain, ytrain_casual)
  clf_2.fit(Xtrain, ytrain_casual)
  clf_3.fit(Xtrain, ytrain_casual)
  clf_4.fit(Xtrain, ytrain_casual)
  clf_5.fit(Xtrain, ytrain_casual)


  print 'Finished fitting'


  dt_casual = clf_1.predict(Xtest)
  ada_casual = clf_2.predict(Xtest)
  grad_casual = clf_3.predict(Xtest)
  rf_casual = clf_4.predict(Xtest)
  et_casual = clf_5.predict(Xtest)


  print "regular decision tree"
  print rmsle(ytest, dt_regular + dt_casual)
  print "boosted decision tree"
  print rmsle(ytest, ada_regular + ada_casual)
  print "gradient tree boosting"
  print rmsle(ytest, grad_regular + grad_casual)
  print "random forest classifier"
  print rmsle(ytest, rf_regular + rf_casual)
  print "extra trees classifier"
  print rmsle(ytest, et_casual + et_regular)

if __name__ == '__main__':
  (X,y1,y2,y3) = import_training_file(sys.argv[1], True)
  decision_tree(X, y1, y2, y3)
