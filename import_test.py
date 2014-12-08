import csv
import sys
import matplotlib.pyplot as plt
import math
import numpy as np
import dateutil.parser as dateparser

from import_train import import_training_file
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer


def import_testing_file(filename, discrete=False):
  # data = genfromtxt(filename, delimiter=',', )

  reader = csv.DictReader(open(filename, 'rU'), delimiter=',')
  #reader.next()
  data = []
  datetime = []
  factorize_fields = ['season', 'holiday', 'workingday', 'weather']
  counter = 0
  for row in reader:
    current_feature_dict = {}
    for f in reader.fieldnames:
      # parse out month
      if f == 'day':
        continue
      if f == 'datetime': 
        datetime.append(row[f]) 
        month = dateparser.parse(row[f]).month
        hour = dateparser.parse(row[f]).hour
        current_feature_dict['month'] = month
        current_feature_dict['hour'] = hour
      elif f not in factorize_fields:
        #print row[f]
        current_feature_dict[f] = float(row[f])
      else:
        current_feature_dict[f] = row[f]

    data.append(current_feature_dict)

  vec = DictVectorizer()
  jumble = vec.fit_transform(data).toarray()  # this messes up ordering....
  feature_names = vec.get_feature_names()

  # correct the ordering
  correct_order = ['month', 'hour', 'Sat?', 'Sun?', 'Mon?', 'Tue?', 'Wed?', 'Thu?', 'Fri?', 'season=1',
                   'season=2', 'season=3', 'season=4', 'holiday=0', 'holiday=1', 'workingday=0', 
                   'workingday=1', 'weather=1', 'weather=2', 'weather=3', 'weather=4', 'temp', 'atemp',
                   'humidity', 'windspeed']
  data = []

  for entry in jumble:
    #current_feature = []
    entry_dict = dict(zip(feature_names, entry))
    current_feature = [entry_dict[k] for k in correct_order]
    data.append(current_feature)

  data = np.array(data)

  #print data.shape
  # orig_n, orig_d = data.shape
  # feature_matrix = np.zeros(shape=(orig_n, orig_d))
  # n,d = feature_matrix.shape

  # for idx, row in enumerate(data):

  #   if discrete:
  #     feature_matrix[idx] = np.rint(row)
  #   else:
  #     feature_matrix[idx] = no_label

  # idx = np.arange(data.shape[0])
  # np.random.seed(42)
  # np.random.shuffle(idx)
  # data = data[idx]

  return data, datetime

def create_submission(datetime, y_total_pred):
  # Write to output file
  with open('submission.csv', 'w') as csvfile:
    fieldnames = ['datetime', 'count']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for idx in xrange(len(y_total_pred)):
      temp_dict = {'datetime' : datetime[idx], 'count' : np.rint(y_total_pred[idx])}
      writer.writerow(temp_dict)

def plot_residuals(train_pred, train_given, test_pred, test_given):
  plt.scatter(test_pred, test_pred - test_given, c='g', s=40, alpha=0.8, label='Testing Residuals')
  plt.scatter(train_pred, train_pred - train_given, c='b', s=40, alpha=0.25, label='Training Residuals')
  plt.plot([0,600],[0,0], c='r')
  plt.xlim(0,600)
  plt.legend()
  plt.title('Residuals vs Bike Share Count')
  plt.xlabel("Predicted count values")
  plt.ylabel("Residuals")
  plt.show()

if __name__ == '__main__':
  X_train, y_total, y_reg, y_casual = import_training_file(sys.argv[1])
  X_test, datetime = import_testing_file(sys.argv[2])

  n, _ = X_train.shape
  nTrain = int(0.5*n)  #training on 50% of the data
  sub_Xtrain = X_train[:nTrain,:]
  sub_ytrain = y_total[:nTrain]
  sub_ytrain_registered = y_reg[:nTrain]
  sub_ytest_registered = y_reg[nTrain:]
  sub_ytrain_casual = y_casual[:nTrain]
  sub_ytest_casual = y_casual[nTrain:]
  sub_Xtest = X_train[nTrain:,:]
  sub_ytest = y_total[nTrain:]

  rf_opt = RandomForestRegressor(bootstrap=True, compute_importances=None,
           criterion='mse', max_depth=None, max_features='auto',
           min_density=None, min_samples_leaf=2, min_samples_split=2,
           n_estimators=1000, n_jobs=1, oob_score=True, random_state=None,
           verbose=0)

  # rf_opt.fit(X_train, y_reg)
  # y_reg_pred = rf_opt.predict(X_test)

  # rf_opt.fit(X_train, y_casual)
  # y_casual_pred = rf_opt.predict(X_test)

  # y_test_pred = y_reg_pred + y_casual_pred

  rf_opt.fit(sub_Xtrain, sub_ytrain)
  sub_y_train_pred = rf_opt.predict(sub_Xtrain)
  #print y_total_pred

  rf_opt.fit(sub_Xtrain, sub_ytrain)
  sub_y_test_pred = rf_opt.predict(sub_Xtest)

  # Write to output file
  # create_submission(datetime, y_test_pred)
  plot_residuals(sub_y_train_pred, sub_ytrain, sub_y_test_pred, sub_ytest)

