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
$$J(\theta) = {{1 \over n} \sum_{i=1}^n[y^ilog(f(x^i))+(1-y^i)log(1-f(x^i))]}$$
We will use gradient descent to minimize the cost function. The gradient w.r.t any parameter can be given by

$${\partial{J(\theta)} \over \partial{\theta_i}} = {1 \over n} \sum_{i=1}^n(f(x^i)-y^i)x^T$$

## Results
```
marks_df = pd.read_csv("marks.txt", header=None)
X = marks_df.iloc[:, :-1]
y = marks_df.iloc[:, -1]

X = np.c_[np.ones((X.shape[0], 1)), X]
y = y[:, np.newaxis]
# Create a matrix for the weights of each attribute
theta = np.zeros((X.shape[1], 1))

log_reg = LogisticRegressionGD()
log_reg.fit(X, y, theta)

x_values = [np.min(X[:, 1] - 5), np.max(X[:, 2] + 5)]
params = log_reg.opt_weights[0]
y_values = - (params[0] + np.dot(params[1], x_values)) / params[2]

admitted = marks_df.loc[y == 1]
not_admitted = marks_df.loc[y == 0]

plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
plt.plot(x_values, y_values, label='Decision Boundary')
plt.xlabel('Marks in 1st Exam')
plt.ylabel('Marks in 2nd Exam')
plt.legend()
plt.show()
```
![image](https://user-images.githubusercontent.com/88197584/227803027-869b8af8-896e-4853-9bb6-1d63d4643663.png)

The articles I relied on to create the class:
 - https://towardsdatascience.com/building-a-logistic-regression-in-python-301d27367c24
 - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
 - https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_tnc.html

P.S The next implementation will add l1, l2 and elasticnet regularization
