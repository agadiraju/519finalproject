# neural nets doesnt really make sense

import math
import numpy as np
import sys

from import_train import import_training_file
from import_train import rmsle
from sklearn.neural_network import BernoulliRBM
from nn import NeuralNet


if __name__ == '__main__':
  (X, y) = import_training_file(sys.argv[1], True)
  hidden_layers = [5]
  learningRate = 1.6
  epsil = 0.12  
  eps = 1000

  neural_network = NeuralNet(hidden_layers, learningRate, epsilon=epsil, numEpochs=eps)
  neural_network.fit(X, y)
  nn_predict = neural_network.predict(X)
