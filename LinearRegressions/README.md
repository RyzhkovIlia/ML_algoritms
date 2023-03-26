# LinearRegressioinGD
## Cost Function
The loss is the error in our predicted value of weights and intercept. Our goal is to minimize this error to obtain the most accurate value of weights and intercept.
We will use the Mean Squared Error function to calculate the loss. There are three steps in this function:

$$E = {{1 \over n} \sum_{i=0}^n{(y_i - (weight*x_i + intercept))^2}}$$

Calculate the partial derivative of the loss function with respect to weights, and plug in the current values of x, y, weights and intercept in it to obtain the derivative value D.

$$D_{weights} = {{-2 \over n} \sum_{i=0}^n{x_i(y_i - y_i^*)}}$$

$$D_{intercept} = {{-2 \over n} \sum_{i=0}^n{(y_i - y_i^*)}}$$

We repeat this process until our loss function is a very small value or ideally 0 (which means 0 error or 100% accuracy). The value of weights and intercept that we are left with now will be the optimum values.

The following data can be taken as an example
```
np.random.seed(42)
x = np.random.rand(100, 1)
y = 2 + 3 * x + np.random.rand(100, 1)
```
## Results
### Penalty = None
```
lin_reg = LinearRegressionGD(penalty=None)
lin_reg.fit(x=x,
    y=y,
    learning_rate=0.01,
    C=5,
    n_iterations=666
    )
prediction = lin_reg.predict(x=x)
plt.scatter(x, y, s=10, c='b')
plt.plot(x, pred, c='r')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```
![image](https://user-images.githubusercontent.com/88197584/227799657-6363d30b-7448-45e5-be70-29e62a610360.png)

### Penalty = l1
```
lin_reg = LinearRegressionGD(penalty='l1')
lin_reg.fit(x=x,
    y=y,
    learning_rate=0.01,
    C=5,
    n_iterations=666
    )
prediction = lin_reg.predict(x=x)
plt.scatter(x, y, s=10, c='b')
plt.plot(x, pred, c='r')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```
![image](https://user-images.githubusercontent.com/88197584/227799765-ed9f1ac2-a1b2-4100-a5ad-9ab3aef9cbd8.png)

### Penalty = l2
```
lin_reg = LinearRegressionGD(penalty='l2')
lin_reg.fit(x=x,
    y=y,
    learning_rate=0.01,
    C=5,
    n_iterations=666
    )
prediction = lin_reg.predict(x=x)
plt.scatter(x, y, s=10, c='b')
plt.plot(x, pred, c='r')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```
![image](https://user-images.githubusercontent.com/88197584/227799830-503d4afa-a848-4799-8ab7-a934962ae27a.png)

### Penalty = elasticnet
```
lin_reg = LinearRegressionGD(penalty='elasticnet')
lin_reg.fit(x=x,
    y=y,
    learning_rate=0.01,
    C=5,
    n_iterations=666
    )
prediction = lin_reg.predict(x=x)
plt.scatter(x, y, s=10, c='b')
plt.plot(x, pred, c='r')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```
![image](https://user-images.githubusercontent.com/88197584/227799856-167063f3-9275-4d49-a065-271553c1bdfc.png)
