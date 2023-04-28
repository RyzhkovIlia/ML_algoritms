# K Nearest Neighbors
The k Nearest Neighbors method (kNN) is also a very popular classification method, also sometimes used in regression problems. It is, along with the decision tree, one of the most understandable approaches to classification. At the level of intuition, the essence of the method is as follows: look at your neighbors, which prevail, that is you. Formally the basis of the method is the compactness hypothesis: if the metric of distance between examples is entered well enough, similar examples are much more likely to be in the same class than in different ones.

The KNN algorithm can be divided into two simple phases: learning and classification. During training, the algorithm simply remembers the observation feature vectors and their class labels (i.e., examples). Also the algorithm parameter k is set, which specifies the number of "neighbors" to be used in classification.

During the classification phase, a new object is presented, for which the class label is not given. For it, k nearest (in the sense of some metric) pre-classified observations are determined. Then the class to which most of the k nearest neighboring examples belong is chosen, and the object being classified belongs to the same class.

## Distance metrics

To determine which data points are closest to a given query point, you need to calculate the distance between the query point and other data points. These distance metrics help form decision boundaries that divide query points into different regions.

1. Euclidean distance (p = 2): this is the most commonly used distance measure, and it is bounded by vectors with valid values. Using the formula below, it measures a straight line between a query point and another measured point.
$$d(x,y) = \sqrt{\sum(y_i-x_i)^2}$$

2. Manhattan distance (p=1): This is another popular distance metric that measures the absolute value between two points. It is also called cab distance or city block distance because it is usually visualized by a grid illustrating how one can travel from one address to another through the streets of the city.
$$Manhattan Distance = d(x,y) = (\sum|x_i-y_i|)$$

3. Minkowski distance: this distance measure is a generalized form of the Euclidean and Manhattan distance metrics. The parameter p in the formula below allows the creation of other distance measures. Euclidean distance is represented by this formula when p equals two, and Manhattan distance is denoted by p equal to one.
$$MinkowskiDistance = (\sum|x_i-y_i|)^{1/p}$$

4. Hamming distance: this method is usually used with Boolean or string vectors, determining the points where the vectors do not overlap. As a result, it is also called the overlap index. This can be represented by the following formula:
$$HammingDistance = D_H = (\sum|x_i-y_i|)$$
* x=y → D=0
* x≠y → D≠1

## Algorithm operation

An important part of the method is **normalization**. Different attributes usually have different ranges of represented values in the sample. For example, attribute **A** is represented in the range from 0.01 to 0.05, and attribute **B** is represented in the range from 500 to 1000. In this case, distance values can be highly dependent on attributes with larger ranges. Therefore, the data in most cases go through normalization.
## Check Regression

```
seed = 42
X, y = make_regression(n_samples=1000,n_features=5)
```

Normalization
```
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
```

## How to use
Model initialization
```
kn_reg = KNNRegressor(n_neighbors=5,
                        p=2)
kn_reg.fit(X=X_train_reg,
            y=y_train_reg)
```
Get Predict
```
predict_reg = kn_reg.predict(X=X_test_reg)
```
Metrics
```
print('MAE', mean_absolute_error(y_test, predict_reg))
print('MSE', mean_squared_error(y_test, predict_reg))
print('RMSE', mean_squared_error(y_test, predict_reg, squared=False))
print('MAPE', mean_absolute_percentage_error(y_test, predict_reg))
print('R2', r2_score(y_test, predict_reg), '\n')
```
Check sklearn model
```
sk_kn_reg = KNeighborsRegressor(n_neighbors=5,
                            p=2)
sk_kn_reg.fit(X=X_train_reg, 
            y=y_train_reg)
prediction_sk_reg = sk_kn_reg.predict(X=X_test_reg)

print('SKLEARN PREDICT')
print('MAE', mean_absolute_error(y_test, prediction_sk_reg))
print('MSE', mean_squared_error(y_test, prediction_sk_reg))
print('RMSE', mean_squared_error(y_test, prediction_sk_reg, squared=False))
print('MAPE', mean_absolute_percentage_error(y_test, prediction_sk_reg))
print('R2', r2_score(y_test, prediction_sk_reg))
```
## Results
```
MY_REGRESSION
MAE = 29.79082061549431
MSE = 1529.9736618306097
RMSE = 39.11487775553708
MAPE = 2.396708419536858
R2 = 0.9109278722208812

SKLEARN PREDICT
MAE = 27.223886988125233
MSE = 1317.9277063785648
RMSE = 36.30327404489249
MAPE = 1.5982146028403488
R2 = 0.9232727804439882
```

## Check Classification

```
seed = 42
X_cl, y_cl = make_classification(n_samples=1000, n_features=5)
```
Normalization
```
scaler = StandardScaler()
X_cl = scaler.fit_transform(X_cl)
X_train_cl, X_test_cl, y_train_cl, y_test_cl = train_test_split(X_cl, y_cl, test_size=0.2, random_state=seed)
```
## How to use
Model initialization
```
kn_model = KNNClassifier(n_neighbors=5,
                        p=2)
kn_model.fit(X=X_train_cl,
            y=y_train_cl)
```
Get Predict
```
predict_cl = kn_model.predict(X=X_test_cl)
```

Metrics
```
print("MY_CLASSIFICATION")
print('precision', precision_score(y_test_cl, predict_cl))
print('recall', recall_score(y_test_cl, predict_cl), '\n')
```
Check sklearn model
```
sk_model_cl = KNeighborsClassifier(n_neighbors=5,
                                p=2)
sk_model_cl.fit(X=X_train_cl, y = y_train_cl)
sk_pred_cl = sk_model_cl.predict(X_test_cl)
print('SKLEARN PREDICT')
print('precision', precision_score(y_test_cl, sk_pred_cl))
print('recall', recall_score(y_test_cl, sk_pred_cl))
```

## Results
```
MY_CLASSIFICATION
precision = 0.8817204301075269
recall = 0.8367346938775511

SKLEARN PREDICT
precision = 0.8863636363636364
recall = 0.7959183673469388
```

The articles I relied on to create the class:
- https://medium.com/analytics-vidhya/implementing-k-nearest-neighbours-knn-without-using-scikit-learn-3905b4decc3c
- https://towardsdatascience.com/create-your-own-k-nearest-neighbors-algorithm-in-python-eb7093fc6339