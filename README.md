519finalproject
===============

#Naive Bayes and Neural Networks

So far I tried Naive Bayes and Neural Networks. They both fail for the same reason -- the # of bikes taken out can be anywhere
from 0 to 997. It's too broad of a classification to make.

=======
# SVMs
* RMSLE sigmoid =  3.99758669678
* RMSLE linear =  2.85748622823
* RMSLE rbf =  3.87239506429
* RMSLE rbf with C=5 gamma=0.0001 = 3.48305326784

=======
RMSLE naiveBayes = 2.16326900532

=======

# Decision Trees (with no max depth)
## Adaboost
### iter = 500
It is 1.25045194287.
### iter = 100
1.26517399599
## Regular
My score is the second lowest: 1.32651746013
## Random Forest Classifier
### iter = 100
My score for this is 1.29524349847
## Extra Trees Classifier
### iter = 100
Score is 1.3031444301

# Logistic Regression
## Results from trying different penalties
* L1 Penalty with C=10.0
  * 2.60695415935
* L2 Penalty with C=10.0
  * 2.6547980647
* L1 Penalty with C=100.0
  * 2.57870120062
* L2 Penalty with C=100.0
  * 2.585980683
* L1 Penalty with C=1000.0
  * 2.53797970939
* L2 Penalty with C=1000.0
  * 2.61280057305
* L1 Penalty with C=10000.0
  * 2.56804227912
* L2 Penalty with C=10000.0
  * 2.57051736226
* L1 Penalty with C=100000.0
  * 2.55627986963
* L2 Penalty with C=100000.0
  * 2.59808068079

## No penalty with CE-5
* 2.70629841642


