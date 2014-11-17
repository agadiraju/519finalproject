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
  feature_matrix = np.zeros(shape=(orig_n, orig_d - 3))
  label_matrix = np.zeros(shape=(orig_n, 1))
  n,d = feature_matrix.shape

  idx = np.arange(n)
  np.random.seed(42)
  np.random.shuffle(idx)
  feature_matrix = feature_matrix[idx]
  label_matrix = label_matrix[idx]

  current_hour = 0
  for idx, row in enumerate(data):
    if current_hour == 24:
      current_hour = 0

    label = row[-1]
    no_label = row[:-3]  # remove label and corresponding counts
    no_label[0] = current_hour  # quick hack to get hour time info
    current_hour += 1

    if discrete:
      feature_matrix[idx] = np.floor(no_label)
    else:
      feature_matrix[idx] = no_label

    label_matrix[idx] = label

  idx = np.arange(n)
  np.random.seed(42)
  np.random.shuffle(idx)
  feature_matrix = feature_matrix[idx]
  label_matrix = label_matrix[idx]

  return (feature_matrix, np.ravel(label_matrix))

if __name__ == '__main__':
  import_training_file(sys.argv[1])
