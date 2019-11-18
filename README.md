# Wine
Choosing the number of neighbors. 
When using the Wine dataset, it is required to predict the grape variety from which wine is made using the results of chemical analyzes.
## Introduction
Metric methods are based on the compactness hypothesis, the essence of which is that objects with similar feature descriptions have similar values ​​of the target variable. If this hypothesis is true, then you can build a forecast for a new object based on objects close to it from the training set - for example, by averaging their answers (for regression) or by choosing the class most popular among them (for classification). Methods of this type are called metric. They have several features:

- The training procedure, in fact, is absent - just remember all the objects of the training sample
- You can use a metric that takes into account the characteristics of a particular data set - for example, the presence of categorical (nominal) features
- With the right choice of metric and a sufficient size of the training sample, metric algorithms show quality close to optimal

Metric methods are sensitive to the scale of signs - so, if the scale of one of the signs significantly exceeds the scale of the other signs, then their values ​​will practically not affect the responses of the algorithm. Therefore, it is important to scale features. This is usually done by subtracting the mean of the characteristic and dividing by the standard deviation.

## Scikit-Learn implementation
The "k" nearest neighbors method is implemented in the "sklearn.neighbors.KNeighborsClassifier" class. The main parameter is "n_neighbors", which sets the number of neighbors to build the forecast.

You will need to do block cross-validation. Cross-validation consists of dividing the sample into m disjoint blocks of approximately the same size, after which m steps are performed. At the i-th step, the i-th block acts as a test sample, the union of all other blocks as a training sample. Accordingly, at each step, the algorithm is trained on some training set, after which its quality is calculated on the test set. After completing m steps, we get m quality indicators, the averaging of which gives an assessment of cross-validation. You can read more about cross-validation in the video "The problem of retraining. Methodology for solving machine learning problems" from the first module, as well as read on Wikipedia or in the scikit-learn documentation.

Technically, cross-validation is carried out in two stages:

- The "sklearn.model_selection.KFold" splitter generator is created, which defines the set of splits for training and validation. The number of blocks in cross validation is determined by the "n_splits" parameter. Please note that the order of objects in the sample may not be random, this may lead to bias in the cross-validation assessment. To eliminate this effect, the sample objects are randomly mixed before being divided into blocks. To mix, just pass the "shuffle = True" parameter to the KFold generator.
- You can calculate the quality on all partitions using the "sklearn.model_selection.cross_val_score" function. The classifier is passed as an "estimator" parameter, the partition generator from the previous step is used as the "cv" parameter. Using the "scoring" parameter, you can specify a measure of quality; by default, the classification tasks use the proportion of correct answers ("accuracy"). The result is an array whose values ​​need to be averaged.

Reduction of signs to the same scale can be done using the "sklearn.preprocessing.scale" function, which requires a matrix of attributes to be input and a scaled matrix in which each column has a zero mean value and a unit standard deviation.

## Prerequisits
- [Python 3.7](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/installing/)

## Installation
```
pip install pandas
pip install numpy
pip install scikit-learn

```
