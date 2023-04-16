from sklearn.datasets import make_regression, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import recall_score, precision_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from KNN import KNNRegressor, KNNClassifier

# Create dataset
seed = 42
X_reg, y_reg = make_regression(n_samples=1000,n_features=5)

#Normalization
scaler = StandardScaler()
X_reg = scaler.fit_transform(X_reg)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=seed)

#Use class KNNRegression
kn_reg = KNNRegressor(n_neighbors=5,
                        p=2)
kn_reg.fit(X=X_train_reg,
            y=y_train_reg)

#Get predict
predict_reg = kn_reg.predict(X=X_test_reg)

#Metrics
print("MY_REGRESSION")
print('MAE', mean_absolute_error(y_test_reg, predict_reg))
print('MSE', mean_squared_error(y_test_reg, predict_reg))
print('RMSE', mean_squared_error(y_test_reg, predict_reg, squared=False))
print('MAPE', mean_absolute_percentage_error(y_test_reg, predict_reg))
print('R2', r2_score(y_test_reg, predict_reg), '\n')

#Check sklearn model
sk_kn_reg = KNeighborsRegressor(n_neighbors=5,
                            p=2)
sk_kn_reg.fit(X=X_train_reg, 
            y=y_train_reg)
prediction_sk_reg = sk_kn_reg.predict(X=X_test_reg)
print('SKLEARN PREDICT')
print('MAE', mean_absolute_error(y_test_reg, prediction_sk_reg))
print('MSE', mean_squared_error(y_test_reg, prediction_sk_reg))
print('RMSE', mean_squared_error(y_test_reg, prediction_sk_reg, squared=False))
print('MAPE', mean_absolute_percentage_error(y_test_reg, prediction_sk_reg))
print('R2', r2_score(y_test_reg, prediction_sk_reg), '\n')
    
# Create dataset
X_cl, y_cl = make_classification(n_samples=1000,n_features=5)

#Normalization
scaler = StandardScaler()
x = scaler.fit_transform(X_cl)
X_train_cl, X_test_cl, y_train_cl, y_test_cl = train_test_split(X_cl, y_cl, test_size=0.2, random_state=seed)

#Use class KNNClassifier
kn_model = KNNClassifier(n_neighbors=5,
                        p=2)
kn_model.fit(X=X_train_cl,
            y=y_train_cl)

#Get Predict
predict_cl = kn_model.predict(X=X_test_cl)

#Metrics
print("MY_CLASSIFICATION")
print('precision', precision_score(y_test_cl, predict_cl))
print('recall', recall_score(y_test_cl, predict_cl), '\n')

#Check sklearn model
sk_model_cl = KNeighborsClassifier(n_neighbors=5,
                                p=2)
sk_model_cl.fit(X=X_train_cl, y = y_train_cl)
sk_pred_cl = sk_model_cl.predict(X_test_cl)
print('SKLEARN PREDICT')
print('precision', precision_score(y_test_cl, sk_pred_cl))
print('recall', recall_score(y_test_cl, sk_pred_cl))