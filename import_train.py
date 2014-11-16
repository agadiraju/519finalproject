import sys
import math
import numpy as np

from numpy import genfromtxt

def rmsle(trainy, predicty):
  metric = 0
  n = len(trainy)
  for idx in xrange(n):
    y_train = trainy[idx]
    y_predict = predicty[idx]

    metric += ((math.log(y_predict) - math.log(y_train)) ** 2)

  return math.sqrt(float(1.0/n) * metric)

def import_training_file(filename, discrete=False):
  data = genfromtxt(filename, delimiter=',', )
  data = data[1:]  # remove header row
  orig_n, orig_d = data.shape
  feature_matrix = np.zeros(shape=(orig_n, orig_d - 2))
  label_matrix = np.zeros(shape=(orig_n, 1))
  
  for idx, row in enumerate(data):
    label = row[-1]
    no_label = row[:-1]  # remove label
    feature_matrix[idx] = no_label[1:]  # remove leading hour time
    if discrete:
      feature_matrix[idx] = np.floor(feature_matrix[idx])
    label_matrix[idx] = label

  return (feature_matrix, np.ravel(label_matrix))

# if __name__ == '__main__':
#   import_training_file(sys.argv[1])