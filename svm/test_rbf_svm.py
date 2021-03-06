print(__doc__)

import numpy as np
from sklearn import metrics
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from datetime import datetime
from import_train import rmsle
from import_train import import_training_file
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import preprocessing as pre
from scipy import sparse
from sklearn.svm import SVR
import sys

if __name__ == '__main__':
	(X, y_total, y_regis, y_casual) = import_training_file(sys.argv[1], True)

	n,d = X.shape
	nTrain = 0.5*n

	Xtrain = X[:nTrain,:]
	y_casual_train = y_casual[:nTrain]
	y_regis_train = y_regis[:nTrain]
	y_total_train = y_total[:nTrain]
	Xtest = X[nTrain:,:]
	y_casual_test = y_casual[nTrain:]
	y_regis_test = y_regis[nTrain:]
	y_total_test = y_total[nTrain:]

	'''
	#rbf
	param_grid = {'C': [1, 5, 10, 100],'gamma': [0.00001,0.0001, 0.001, 0.01, 0.1],}
	#clf = GridSearchCV(SVC(kernel='rbf'), param_grid,n_jobs=-1)
	clf = SVC(kernel='rbf',C=5.0,gamma=0.0001)
	clf.fit(Xtrain,ytrain)
	pred = clf.predict(Xtest)
	
	print "best estimator = ",clf.best_estimator_
	print "RMSE rbf = ", rmsle(ytest, pred)
	#print classification_report(ytest, pred)
	'''
	#new stuff
	clf_regis = SVR(kernel='rbf')
	clf_regis.fit(Xtrain,y_regis_train)
	pred_regis = clf_regis.predict(Xtest)
	
	clf_casual = SVR(kernel='rbf')
	clf_casual.fit(Xtrain,y_casual_train)
	pred_casual = clf_casual.predict(Xtest)

	pred_total = pred_casual + pred_regis
	print "RMSLE rbf total = ", rmsle(y_total_test, pred_total)