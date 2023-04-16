# GaussianNaiveBayes
A naive Bayesian algorithm is a probabilistic machine learning algorithm based on the application of Bayes theorem and used in a variety of classification problems. In this article, we will review all the basic principles and concepts associated with the naive Bayesian algorithm, so that you won't have any problems understanding it.
## Bayes' Theorem
Bayes' Theorem is a simple mathematical formula used to calculate conditional probabilities.
Conditional probability is the probability of one event occurring when another event (whether by assumption, supposition, proven or unproven assertion) has already occurred.
The formula for determining conditional probability:
$$P(A|B) = {P(B|A)*P(A) \over {P(B)}}$$
- P(A | B) is the a posteriori probability (that A of B is true)
- P(A) is the a priori probability (independent probability of A)
- P(B | A) is the probability of a given trait value at a given class. (That B of A is true)
- P(B) is the a priori probability at our trait value.(independent probability of B)

This Bayes theorem can now be used to create the following classification model:

$$P(y|x_1, x_2, ..x_N) = {P(x_1|y)*P(x_2|y)*..P(x_N|y) * P(y) \over {P(x_1)*P(x_2)*..P(x_N)}}$$

Now you can get the values for each of the parameters: just take the data and substitute them into the equation. For all combinations of these parameters, the denominator does not change, it remains static. So you can discard it by introducing proportionality into the equation:

$$P(y|x_1, x_2, ..., x_N) \propto {P(y)\prod_{i=1}^NP(x_i|y)}$$

There are cases when the classification can be multivariate. So we have to find the class variable (y) with maximum probability.

$$y = {argmax_iP(y)\prod_{i=1}^NP(x_i|y)}$$

Using this function, you can get a class based on the available predictors/parameters.

The posterior probability P(y|X) is calculated as follows: first, a frequency table is created for each parameter relative to the desired result. Then the likelihood tables are generated from the frequency tables, and then the posterior probability for each class is calculated using Bayes equation. The class with the highest posterior probability will be the predicted result.

### Important!
Gaussian distribution: continuous values of all characteristics are assumed to have a Gaussian distribution (normal distribution). When plotted, a bell-shaped curve is obtained, which is symmetrical with respect to the mean values of the characteristics.
It is assumed that we have a Gaussian probability of characteristics, so the conditional probability will be determined as follows:

$$P(x) = { \exp^{-{(x-\mu)^2 \over {2*\sigma^2}}} \over {\sqrt{2*\pi * \sigma^2}}}$$

## Data
The following data can be taken as an example
```
X  = load_iris().data
y = load_iris().target
data = pd.concat([pd.DataFrame(X, columns=['featuere_'+str(i) for i in range(1, X.shape[1]+1)]), 
                pd.Series(y, name='target')], 
                axis=1)
X = data.iloc[:, :-1]
y = data['target']
```
## How to use
Model initialization
```
mod = GaussianNaiveBayes()
mod.fit(
    X=X, 
    y=y
    )
```
Get Predict
```
pred_gaus = mod.predict(X)
```
Metrics
```
print('precision', precision_score(y, pred_gaus, average='macro'))
print('recall', recall_score(y, pred_gaus, average='macro'), '\n')
```
Check sklearn model
```
sk_model = GaussianNB()
sk_model.fit(
    X=X,
    y=y
    )
pred_sk = sk_model.predict(X=X)
print('SKLEARN PREDICT')
print('precision', precision_score(y, pred_sk, average='macro'))
print('recall', recall_score(y, pred_sk, average='macro'))
```
## Results
```
precision 0.96
recall 0.96 

SKLEARN PREDICT
precision 0.96
recall 0.96
```

The articles I relied on to create the class:
 - https://nuancesprog.ru/p/10732/
 - https://medium.com/@rangavamsi5/na%C3%AFve-bayes-algorithm-implementation-from-scratch-in-python-7b2cc39268b9
