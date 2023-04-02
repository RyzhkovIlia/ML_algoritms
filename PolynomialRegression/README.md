# PolynomialRegressionGD
In this folder, i will look at the Polynomial Regression algorithm which can be used to fit non-linear data by modifying the hypothesis function and by adding new features we want to add to the input.
Polynomial Regression is just another version of the standard Linear Regression.
In linear regression with one predictor, we have the following equation:
$$Y = {bias + w * x}$$

This linear equation can be used to represent a linear relationship. But in polynomial regression we have a polynomial equation of degree n , represented as:
$$Y = {bias + w_1 * x + w_2 * x^2 + w_3 * x^3 + ... + w_N * x^N}$$

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
seed = 42
np.random.seed(seed)
x = np.random.rand(1000,5)
y = 5*((x[:, 1].reshape((x.shape[0], 1)))**(2)) + np.random.rand(1000,1)
```
## How to use
Model initialization
```
lin_reg = PolynomialRegressionGD(penalty='l1',
                            random_state=seed,
                            plot_loss=True)
lin_reg.fit(X=x,
            y=y,
            learning_rate=0.01,
            C=0.0001,
            max_n_iterations=10000,
            batch_size=128,
            degree=[2]
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
print('MAPE', mean_absolute_percentage_error(y, prediction))
print('R2', r2_score(y, prediction), '\n')
```
Check sklearn model
```
sk_lin = LinearRegression()
x_copy = x.copy()
poly = PolynomialFeatures(2)
x_copy =poly.fit_transform(x_copy)
sk_lin.fit(X=x_copy, 
            y=y)
prediction_sk = sk_lin.predict(X=x_copy)
print('SKLEARN PREDICT')
print('MAE', mean_absolute_error(y, prediction_sk))
print('MSE', mean_squared_error(y, prediction_sk))
print('RMSE', mean_squared_error(y, prediction_sk, squared=False))
print('MAPE', mean_absolute_percentage_error(y, prediction_sk))
```
## Results
```
MAE 0.2642696930369943
MSE 0.09883823649640755
RMSE 0.3143854902765195
MAPE 0.2974067548498449
R2 0.958454781036168

SKLEARN PREDICT
MAE 0.2463234881082973
MSE 0.08193473073072353
RMSE 0.2862424334907799
MAPE 0.35510135457299064
R2 0.965559924482523
```
![image](https://user-images.githubusercontent.com/88197584/229350603-be91f9bf-5e83-4859-aed6-8cf680da3b1a.png)

The articles I relied on to create the class:
 - https://towardsdatascience.com/polynomial-regression-in-python-b69ab7df6105
 - https://morioh.com/p/28d4fa379f60
