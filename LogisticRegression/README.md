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

$${\partial{J(\theta)} \over \partial{\theta_i}} = {-2 \over n} \sum_{i=1}^n(f(x^i)-y^i)x^T$$

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
log_reg = LogisticRegressionGD(penalty='l2',
                                random_state=42,
                                plot_loss=True) # l1, l2, elasticnet
log_reg.fit(X = x_train, 
            y = y_train, 
            C = 0.01, 
            learning_rate=0.02, 
            max_n_iterations=1000)
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
sk_model = LogisticRegression(C = 0.01, 
                                max_iter=1000, 
                                random_state=42)
sk_model.fit(X=x_train, y = y_train)
sk_pred = sk_model.predict(x_test)
print('SKLEARN PREDICT')
print('precision', precision_score(y_test, sk_pred))
print('recall', recall_score(y_test, sk_pred))
```
## Results
```
precision 0.5274725274725275
recall 0.8727272727272727 

SKLEARN PREDICT
precision 0.7894736842105263
recall 0.5454545454545454
```

![image](https://user-images.githubusercontent.com/88197584/228609433-f490ddf5-d211-447e-9566-e0528900fe51.png)

The articles I relied on to create the class:
 - https://towardsdatascience.com/building-a-logistic-regression-in-python-301d27367c24
 - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
 - https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_tnc.html
