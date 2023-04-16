import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from GaussianNaiveBayes import GaussianNaiveBayes
from sklearn.metrics import recall_score, precision_score
from sklearn.naive_bayes import GaussianNB

# Create dataset
seed = 42
X_cl, y_cl = make_classification(n_samples=1000,n_features=5)
df_cl = pd.DataFrame(X_cl, columns=['feat_'+str(i) for i in range(X_cl.shape[1])])
y_cl = pd.Series(y_cl)
X_train_cl, X_test_cl, y_train_cl, y_test_cl = train_test_split(df_cl, y_cl, test_size=0.2, random_state=seed)

# #Use class GaussianNaiveBayes
mod = GaussianNaiveBayes()
mod.fit(
    X=X_train_cl, 
    y=y_train_cl
    )
pred_gaus = mod.predict(X_test_cl)

#Metrics
print('precision', precision_score(y_test_cl, pred_gaus))
print('recall', recall_score(y_test_cl, pred_gaus), '\n')

#Check sklearn model
sk_model = GaussianNB()
sk_model.fit(
    X=X_train_cl,
    y=y_train_cl
    )
pred_sk = sk_model.predict(X=X_test_cl)

#Metrics
print('SKLEARN PREDICT')
print('precision', precision_score(y_test_cl, pred_sk))
print('recall', recall_score(y_test_cl, pred_sk))