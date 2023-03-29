# LinearRegressionGD
## Cost Function
The loss is the error in our predicted value of weights and intercept. Our goal is to minimize this error to obtain the most accurate value of weights and intercept.
We will use the Mean Squared Error function to calculate the loss. There are three steps in this function:

$$E = {{1 \over n} \sum_{i=0}^n{(y_i - (weight*x_i + intercept))^2}}$$

Calculate the partial derivative of the loss function with respect to weights, and plug in the current values of x, y, weights and intercept in it to obtain the derivative value D.

$$D_{weights} = {{-2 \over n} \sum_{i=0}^n{x_i(y_i - y_i^*)}}$$

$$D_{intercept} = {{-2 \over n} \sum_{i=0}^n{(y_i - y_i^*)}}$$

If one of the available regularizations is used, the regularization terms are added to the cost function.
We repeat this process until our loss function is a very small value or ideally 0 (which means 0 error or 100% accuracy). The value of weights and intercept that we are left with now will be the optimum values.

The following data can be taken as an example
```
x = np.random.rand(1000, 10)
y = 2 + 3*x[:, 0].reshape((1000, 1))**2 + np.random.rand(1000, 1))
```
## How to use
Model initialization
```
lin_reg = LinearRegressionGD(penalty='elasticnet',
                            random_state=42,
                            plot_loss=True)
lin_reg.fit(X=x,
            y=y,
            learning_rate=0.1,
            C=0.0001,
            max_n_iterations=10000
            )
```
Get Predict
```
prediction = lin_reg.predict(X=x)
```
Metrics
```
print('MAE', mean_absolute_error(y, prediction))
print('MSE', mean_squared_error(y, prediction))
print('RMSE', mean_squared_error(y, prediction, squared=False))
print('MAPE', mean_absolute_percentage_error(y, prediction), '\n')
```
Check sklearn model
```
sk_lin = LinearRegression()
sk_lin.fit(X=x, 
            y=y)
prediction_sk = sk_lin.predict(X=x)
print('SKLEARN PREDICT')
print('MAE', mean_absolute_error(y, prediction_sk))
print('MSE', mean_squared_error(y, prediction_sk))
print('RMSE', mean_squared_error(y, prediction_sk, squared=False))
print('MAPE', mean_absolute_percentage_error(y, prediction_sk))
```
## Results
```
MAE 0.3121887974679742
RMSE 0.3826254095990843
MAPE 0.09538290956625316

SKLEARN PREDICT
MAE 0.3015480403057931
MSE 0.13252664637650863
RMSE 0.3640420942370657
MAPE 0.09227202516334462
```
![image](https://user-images.githubusercontent.com/88197584/228607062-f02b7086-5aec-4021-84b6-824229f7b948.png)

The articles I relied on to create the class:
 - https://towardsdatascience.com/linear-regression-using-python-b136c91bf0a2
 - https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931
 - https://www.geeksforgeeks.org/implementation-of-lasso-regression-from-scratch-using-python/
