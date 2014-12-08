import csv
import sys
import matplotlib.pyplot as plt
import math
import numpy as np
import dateutil.parser as dateparser

from sklearn.feature_extraction import DictVectorizer

def rmsle(trainy, predicty):
  metric = 0
  n = len(trainy)
  for idx in xrange(n):
    y_train = trainy[idx] + .0001
    y_predict = predicty[idx] + .0001

    if y_predict <= 0:
      y_predict = .0001
    # print y_predict
    # print y_train
    # raw_input()
    metric += ((math.log(y_predict) - math.log(y_train)) ** 2)

  return math.sqrt(float(1.0/n) * metric)

def plot_ride_heatmap(filename):
  reader = csv.DictReader(open(filename, 'rU'), delimiter = ',')
  ride_data = np.zeros(shape = (7, 24))  # days x hours

  for row in reader:
    hour = dateparser.parse(row['datetime']).hour
    day = int(row['day']) - 1
    total_rides = float(row['count'])
    ride_data[day][hour] += total_rides

  day_labels = ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
  plt.pcolor(ride_data,cmap=plt.cm.Blues,edgecolors='k')
  plt.xticks(np.arange(0, 24))
  plt.yticks(np.arange(0, 7) + 0.5, day_labels)
  plt.xlabel('Hour')
  plt.ylabel('Day of the Week')
  plt.title('Heatmap of Bike Rides At Specific Hours')
  plt.show()

def import_training_file(filename, discrete=False):
  # data = genfromtxt(filename, delimiter=',', )

  reader = csv.DictReader(open(filename, 'rU'), delimiter=',')
  #reader.next()
  data = []
  factorize_fields = ['season', 'holiday', 'workingday', 'weather']
  counter = 0
  for row in reader:
    current_feature_dict = {}
    for f in reader.fieldnames:
      # parse out month
      if f == 'day':
        continue
      if f == 'datetime':  
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
                   'humidity', 'windspeed', 'casual', 'registered', 'count']
  data = []

  for entry in jumble:
    #current_feature = []
    entry_dict = dict(zip(feature_names, entry))
    current_feature = [entry_dict[k] for k in correct_order]
    data.append(current_feature)

  data = np.array(data)

  orig_n, orig_d = data.shape
  feature_matrix = np.zeros(shape=(orig_n, orig_d - 3))
  total_matrix = np.zeros(shape=(orig_n, 1))
  registered_matrix = np.zeros(shape=(orig_n, 1))
  casual_matrix = np.zeros(shape=(orig_n, 1))
  n,d = feature_matrix.shape

  for idx, row in enumerate(data):

    total = row[-1]
    registered = row[-2]
    casual = row[-3]

    no_label = row[:-3]  # remove label and corresponding counts

    if discrete:
      feature_matrix[idx] = np.rint(no_label)
    else:
      feature_matrix[idx] = no_label

    total_matrix[idx] = total
    registered_matrix[idx] = registered
    casual_matrix[idx] = casual

  idx = np.arange(n)
  np.random.seed(42)
  np.random.shuffle(idx)
  feature_matrix = feature_matrix[idx]
  total_matrix = total_matrix[idx]
  registered_matrix = registered_matrix[idx]
  casual_matrix = casual_matrix[idx]

  return (feature_matrix, np.ravel(total_matrix), np.ravel(registered_matrix), np.ravel(casual_matrix))

if __name__ == '__main__':
  print import_training_file(sys.argv[1])
  #plot_ride_heatmap(sys.argv[1])
