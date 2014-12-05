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
import sys

if __name__ == '__main__':
	(X, y) = import_training_file(sys.argv[1], True)

	n,d = X.shape
	nTrain = 0.5*n 

	# shuffle the data
	#idx = np.arange(n)
	#np.random.seed(42)
	#np.random.shuffle(idx)
	#X = X[idx]
	#y = y[idx]

	Xtrain = X[:nTrain,:]
	ytrain = y[:nTrain]
	Xtest = X[nTrain:,:]
	ytest = y[nTrain:]
	
	#linear
	param_grid = {'C': [1, 5, 10, 100],}
	clf = GridSearchCV(SVC(kernel='linear'), param_grid,n_jobs=-1)
	#clf = SVC(kernel='linear')
	clf.fit(Xtrain,ytrain)
	pred = clf.predict(Xtest)
	print "best estimator = ",clf.best_estimator_
	print "RMSE linear = ", rmsle(ytest, pred)
	