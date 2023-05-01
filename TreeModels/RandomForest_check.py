from sklearn.metrics import recall_score, precision_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_regression, make_classification
from RandomForest import MyRandomForestRegressor, MyRandomForestClassifier

# Create dataset
seed = 42
X_cl, y_cl = make_classification(n_samples=1000,n_features=5)

X_train_cl, X_test_cl, y_train_cl, y_test_cl = train_test_split(X_cl, y_cl, test_size=0.2, random_state=seed)

#Use class DecisionTreeClass
classifier = MyRandomForestClassifier(n_estimators=5, min_samples_split=4, max_depth=5)
classifier.fit(X=X_train_cl,y=y_train_cl)
pred = classifier.predict(X = X_test_cl)

print("MY_CLASSIFICATION")
print('precision', precision_score(y_test_cl, pred))
print('recall', recall_score(y_test_cl, pred), '\n')

#Check sklearn model
sk_model = RandomForestClassifier(n_estimators=5, min_samples_split=4, max_depth=5)
sk_model.fit(X=X_train_cl, y=y_train_cl)
sk_pred_class = sk_model.predict(X=X_test_cl)
print('SKLEARN PREDICT')
print('precision', precision_score(y_test_cl, sk_pred_class))
print('recall', recall_score(y_test_cl, sk_pred_class), '\n')

# Create dataset
X, y = make_regression(n_samples=1000,n_features=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

#Use class DecisionTreeReg
dt_reg = MyRandomForestRegressor(n_estimators=5, min_samples_split=4, max_depth=5)
dt_reg.fit(X=X_train, y=y_train)
pred_reg = dt_reg.predict(X=X_test)

#Metrics
print("MY_REGRESSION")
print('MAE', mean_absolute_error(y_test, pred_reg))
print('MSE', mean_squared_error(y_test, pred_reg))
print('RMSE', mean_squared_error(y_test, pred_reg, squared=False))
print('MAPE', mean_absolute_percentage_error(y_test, pred_reg))
print('R2', r2_score(y_test, pred_reg), '\n')

#Check sklearn model
sk_lin = RandomForestRegressor(n_estimators=5, min_samples_split=4, max_depth=5)
sk_lin.fit(X=X_train, y=y_train)
sk_pred_reg = sk_lin.predict(X=X_test)
print('SKLEARN PREDICT')
print('MAE', mean_absolute_error(y_test, sk_pred_reg))
print('MSE', mean_squared_error(y_test, sk_pred_reg))
print('RMSE', mean_squared_error(y_test, sk_pred_reg, squared=False))
print('MAPE', mean_absolute_percentage_error(y_test, sk_pred_reg))
print('R2', r2_score(y_test, sk_pred_reg))