from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from GradientBoosting import GradientBoostingRegression
from sklearn.ensemble import GradientBoostingRegressor

# Create dataset
seed = 42
X, y = make_regression(n_samples=1000,n_features=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

#Use class GradientBoostingRegression
gb_reg = GradientBoostingRegression(learning_rate=0.1, max_depth=5, min_samples_split=4)
gb_reg.fit(X=X_train, y=y_train, max_trees=50)
gb_pred = gb_reg.predict(X_test)

#Metrics
print("MY_REGRESSION")
print('MAE', mean_absolute_error(y_test,gb_pred))
print('RMSE', mean_squared_error(y_test,gb_pred, squared=False))
print('MAPE',mean_absolute_percentage_error(y_test,gb_pred))
print('R2', r2_score(y_test,gb_pred), '\n')

#Check sklearn model
sk_gb_reg = GradientBoostingRegressor(learning_rate=0.1, max_depth=5, min_samples_split=4)
sk_gb_reg.fit(X=X_train, y=y_train)
prediction_sk = sk_gb_reg.predict(X=X_test)
print('SKLEARN PREDICT')
print('MAE', mean_absolute_error(y_test, prediction_sk))
print('MSE', mean_squared_error(y_test, prediction_sk))
print('RMSE', mean_squared_error(y_test, prediction_sk, squared=False))
print('MAPE', mean_absolute_percentage_error(y_test, prediction_sk))
print('R2', r2_score(y_test, prediction_sk), '\n')