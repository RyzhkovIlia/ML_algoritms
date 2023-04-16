# Linear, Lasso, Ridge and Elasticnet Regression GD
## Cost Function
The loss is the error in our predicted value of weights and intercept. Our goal is to minimize this error to obtain the most accurate value of weights and intercept.
We will use the Mean Squared Error function to calculate the loss. There are three steps in this function:

$$E = {{1 \over n} \sum_{i=0}^n{(y_i - (weight*x_i + bias))^2}}$$

Calculate the partial derivative of the loss function with respect to weights, and plug in the current values of x, y, weights and intercept in it to obtain the derivative value D.

For Linear
$$D_{weights} = {{-2 \over n} \sum_{i=0}^n{x_i(y_i - \hat{y_i})}}$$

For Lasso
$$D_{weights} = {{-2 \over n} \sum_{i=0}^n{x_i(y_i - \hat{y_i})+\lambda |weight|}}$$

For Ridge
$$D_{weights} = {{-2 \over n} \sum_{i=0}^n{x_i(y_i - \hat{y_i})+ \lambda * 2 * weight}}$$

For Elasticenet
$$D_{weights} = {{-2 \over n} \sum_{i=0}^n{x_i(y_i - \hat{y_i})+\lambda |weight| + \lambda * 2 * weight}}$$

$$D_{bias} = {{-2 \over n} \sum_{i=0}^n{(y_i - \hat{y_i})}}$$

If one of the available regularizations is used, the regularization terms are added to the cost function.
We repeat this process until our loss function is a very small value or ideally 0 (which means 0 error or 100% accuracy). The value of weights and intercept that we are left with now will be the optimum values.

The following data can be taken as an example
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
lin_model = Lasso{Ridge, Elasticnet, Linear}RegressionGD(
    random_state=seed,
    plot_loss=True
    )
lin_model.fit(
    X=X_train,
    y=y_train,
    learning_rate=0.01,
    C=0.001,
    max_n_iterations=200,
    batch_size=128
    )
```
Get Predict
```
prediction = lin_model.predict(X=X_test)
```
Metrics
```
print('MAE', mean_absolute_error(y_test, prediction))
print('MSE', mean_squared_error(y_test, prediction))
print('RMSE', mean_squared_error(y_test, prediction, squared=False))
print('MAPE', mean_absolute_percentage_error(y_test, prediction))
print('R2', r2_score(y_test, prediction), '\n')
```
Check sklearn model
```
sk_lin = Lasso{Ridge, Elasticnet, Linear}(
    alpha=0.0001,
    random_state=seed,
    max_iter=200
    )
sk_lin.fit(
    X=X_train, 
    y=y_train
    )
prediction_sk = sk_lin.predict(X=x)
print('SKLEARN PREDICT')
print('MAE', mean_absolute_error(y_test, prediction_sk))
print('MSE', mean_squared_error(y_test, prediction_sk))
print('RMSE', mean_squared_error(y_test, prediction_sk, squared=False))
print('MAPE', mean_absolute_percentage_error(y_test, prediction_sk))
print('R2', r2_score(y_test, prediction_sk))
```

The articles I relied on to create the class:
 - https://towardsdatascience.com/linear-regression-using-python-b136c91bf0a2
 - https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931
 - https://www.geeksforgeeks.org/implementation-of-lasso-regression-from-scratch-using-python/
 - https://towardsdatascience.com/polynomial-regression-in-python-b69ab7df6105
 - https://morioh.com/p/28d4fa379f60


# LogisticRegressionGD
## Cost function
A Linear Regression model can be represented by the equation.
$$f(x) = {\theta^Tx}$$
We then apply the sigmoid function to the output of the linear regression
$$f(x) = {\sigma(\theta^Tx)}$$
where the sigmoid function is represented by,
$$\sigma(t) = {1 \over {1 + \exp^{-t}}}$$
The hypothesis for logistic regression then becomes,
$$f(x) = {1 \over {1 + \exp^{-\theta^Tx}}}$$
If the weighted sum of inputs is greater than zero, the predicted class is 1 and vice-versa. So the decision boundary separating both the classes can be found by setting the weighted sum of inputs to 0.

Like Linear Regression, we will define a cost function for our model and the objective will be to minimize the cost.

If the actual class is 1 and the model predicts 0, we should highly penalize it and vice-versa. As you can see from the below picture, for the plot -log(h(x)) as h(x) approaches 1, the cost is 0 and as h(x) nears 0, the cost is infinity(that is we penalize the model heavily). Similarly for the plot -log(1-h(x)) when the actual value is 0 and the model predicts 0, the cost is 0 and the cost becomes infinity as h(x) approaches 1.

We will use BinaryCrossEntropy
$$J(\theta) = {-{1 \over n} \sum_{i=1}^n[y^ilog(f(x^i))+(1-y^i)log(1-f(x^i))]}$$
We will use gradient descent to minimize the cost function. The gradient w.r.t any parameter can be given by

$${\partial{J(\theta)} \over \partial{\theta_i}} = {-1 \over n} \sum_{i=1}^n(f(x^i)-y^i)x^T$$

## How to use
Read dataset
```
data = pd.read_csv('data.csv', header=None)
x, y = data.iloc[:, :-1], data.iloc[:, -1]
```
Normalization
```
scaler = StandardScaler()
x = scaler.fit_transform(x)
```
Train-test split
```
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```
Model initialization
```
log_reg = LogisticRegressionGD(
    penalty='l2',
    random_state=42,
    plot_loss=True
    ) # l1, l2, elasticnet
log_reg.fit(
    X = x_train, 
    y = y_train, 
    C = 0.01, 
    learning_rate=0.01, 
    max_n_iterations=2000
    )
```
Find optimal threshhold
```
precisions, recalls, thresholds = precision_recall_curve(y_train, log_reg.predict(x_train))
f_scores = np.nan_to_num((2*precisions*recalls)/(precisions+recalls+0.0001))
f_max_index = np.argmax(f_scores)
custom_threshold = thresholds[f_max_index]
```
get predict
```
pred = log_reg.predict(X = x_test)
pred = (pred>custom_threshold).astype(int)
```
Metrics
```
print('precision', precision_score(y_test, pred))
print('recall', recall_score(y_test, pred), '\n')
```
Check sklearn model
```
sk_model = LogisticRegression(
    C = 0.01, 
    max_iter=1000, 
    random_state=42,
    penalty='l2'
    )
sk_model.fit(X=x_train, y = y_train)
sk_pred = sk_model.predict(x_test)
print('SKLEARN PREDICT')
print('precision', precision_score(y_test, sk_pred))
print('recall', recall_score(y_test, sk_pred))
```
## Results
```
precision 0.978494623655914
recall 0.91

SKLEARN PREDICT
precision 1.0
recall 0.83
```

The articles I relied on to create the class:
 - https://towardsdatascience.com/building-a-logistic-regression-in-python-301d27367c24
 - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
 - https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_tnc.html
