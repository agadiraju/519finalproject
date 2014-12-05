print(__doc__)

import numpy as np
from sklearn import metrics
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from datetime import datetime
from import_train import rmsle
from import_train import import_training_file
import sys

if __name__ == '__main__':
	(X, y) = import_training_file(sys.argv[1], True)

	n,d = X.shape
	nTrain = 0.5*n 

	# shuffle the data
	#idx = np.arange(n)
	#np.random.seed(42)
	#np.random.shuffle(idx)
	#y = y[idx]
	#X = X[idx]

	# split the data
	Xtrain = X[:nTrain,:]
	ytrain = y[:nTrain]
	Xtest = X[nTrain:,:]
	ytest = y[nTrain:]
	
	#linear
	clf = SVC(kernel='linear')
	clf.fit(Xtrain,ytrain)
	pred = clf.predict(Xtest)
	print "RMSE linear = ", rmsle(ytest, pred)

	#polynomial
	clf = SVC(kernel='poly')
	clf.fit(Xtrain,ytrain)
	pred = clf.predict(Xtest)
	print "RMSE poly = ", rmsle(ytest, pred)

	#rbf
	clf = SVC(kernel='rbf')
	clf.fit(Xtrain,ytrain)
	pred = clf.predict(Xtest)
	print "RMSE rbf = ", rmsle(ytest, pred)

	#sigmoid
	clf = SVC(kernel='sigmoid')
	clf.fit(Xtrain,ytrain)
	pred = clf.predict(Xtest)
	print "RMSE sigmoid = ", rmsle(ytest, pred)
