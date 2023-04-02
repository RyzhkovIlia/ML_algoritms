import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
class RidgeRegressionGD:
    """Ridge Regression Using Gradient Descent.
    Parameters
    ----------
    random_state (int, optional): _description_. Defaults to None.
    Attributes
    ----------
    coef_ : weights after fitting the model
    intercept_ : free member of regression
    """

    def __init__(self,
                penalty:str=None,
                random_state:int=None,
                plot_loss:bool=False):
        assert\
            (isinstance(random_state, int))|(random_state is None),\
            f'N_iterations must be only integer and > 0. Receive {type(random_state)} = {random_state}.'
        assert\
            isinstance(plot_loss, bool),\
            f'plot_loss must be only bool. Receive {type(plot_loss)} = {plot_loss}.'
        
        self.__random_state = random_state
        self.__plot_loss = plot_loss

        np.random.seed(seed=self.__random_state)

    def __check_params(self, 
                        X):
        """Check corrections parameters
        Args:
            X: array-like, shape = [n_features, n_samples]
        """
        assert\
            (isinstance(self.__learning_rate, float))|(isinstance(self.__learning_rate, int))&(self.__learning_rate>0)&(self.__learning_rate<=1),\
            f'Learning_rate must be only in interval (0, 1]. Receive {self.__learning_rate}.'
        assert\
            isinstance(self.__n_iterations, int),\
            f'N_iterations must be only integer. Receive {type(self.__n_iterations)}.'
        assert\
            isinstance(self.__C, float)|isinstance(self.__C, int),\
            f'C must be only integer or float. Receive {type(self.__C)}.'
        assert\
            (X.shape[0]>0)&(X.shape[1]>0),\
            f'X must not be empty.'
        assert\
            (isinstance(self.__batch_size, int))|(self.__batch_size is None),\
            f'Batch_size must be only integer or None and >0. Receive {self.__batch_size}.'

    def __calculate_gradient(self, 
                            X):
        """Function for find out weight and bias gradients
        Args:
            X: array-like, shape = [n_features, n_samples]
        """
        # If the usual linear regression we find the gradient, lasso, ridge and elastic
        self.__dW = -2*((np.dot(X, self.__residuals))+(2*self.__C*np.sum(self.coef_)))/self.__m
        # Find the gradient for the free term bias
        self.__db = -2*np.sum(self.__residuals)/self.__m
    
    def __update_weights(self, 
                            X, 
                            y):
        """Update weights and bias
        Args:
            X: array-like, shape = [n_features, n_samples]
            y: array-like, shape = [n_samples, n_target_values]
        """
        # Get result
        y_pred = self.intercept_ + np.dot(self.coef_.T, X)
        y_pred = y_pred.reshape((y_pred.shape[1],1))

        # Deviation from the true value
        self.__residuals = y - y_pred

        cost = np.sum(self.__residuals ** 2)/self.__m
        self.cost_list.append(cost)

        # Stop condition
        if (len(self.cost_list)>2):
            self.__flag = False if np.sum(self.__residuals) < -10e30 or (((self.cost_list[-2]/cost)-1)*10000)<1 else True
        else:
            pass

        # Calculate gradients  
        self.__calculate_gradient(X=X)
        gradients = {"derivative_weight": self.__dW,"derivative_bias": self.__db}

        # Update the weights with the gradient found from the production MSE
        self.coef_ -= (self.__learning_rate*gradients["derivative_weight"])

        # Update free member b
        self.intercept_ -= (self.__learning_rate * gradients["derivative_bias"])

    def __plot_cost(self):
        """Show loss curve
        """
        plt.plot(range(len(self.cost_list)), self.cost_list)
        plt.xticks(range(len(self.cost_list)), rotation='vertical')
        plt.xlabel("Number of Iteration")
        plt.ylabel("Cost")
        plt.show()

    def __initialize_weights_and_bias(self,
                                        X)->tuple:
        """Create a matrix for the weights of each attribute and the free member bias
        Args:
            X: array-like, shape = [n_features, n_samples]
        Returns:
            tuple: Weights and bias matrixs
        """

        weights = np.random.rand(X.shape[0], 1)
        bias = np.random.random()
        return (weights, bias)

    def fit(self, 
            X, 
            y,
            batch_size:int=None,
            learning_rate:float=0.001,
            C:float=1.0,
            max_n_iterations:int=1000,
            ):
        """Fit the training data
        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
            Training samples
        y: array-like, shape = [n_samples, 1]
            Target values
        learning_rate: float, learning rate coeff
        C: float , Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
        max_n_iterations: int, count of inerations
        Returns
        -------
        self : object
        """
        self.__learning_rate = learning_rate
        self.__n_iterations = max_n_iterations
        self.__C = C
        self.__flag = True
        self.__batch_size = batch_size
        self.cost_list = []

        # Check correct params
        self.__check_params(X=X)

        X = X.T
        y = np.array(y)
        assert\
            (y.shape[0]==X.shape[1]),\
            f'Y shape must be equal X shape.'
        y = y.reshape((y.shape[0], 1))

        # Create a matrix for the weights of each attribute and free member b
        self.coef_, self.intercept_ = self.__initialize_weights_and_bias(X=X)

        # Dimensionality of features and number of training examples
        self.n_features_in_, self.__m = X.shape

        for _ in range(self.__n_iterations):
            if self.__flag:
                if self.__batch_size is not None:
                    for i in range((self.__m-1)//self.__batch_size + 1):
                        # Defining batches.
                        start_i = i*self.__batch_size
                        end_i = start_i + self.__batch_size
                        xb = X[:,start_i:end_i]
                        yb = y[start_i:end_i]
                        # Updating the weights
                        self.__update_weights(X=xb,
                                        y=yb)
                else:
                    # Updating the weights
                    self.__update_weights(X=X,
                                        y=y)
            else:
                break
        if self.__plot_loss:
            #Plot
            self.__plot_cost()
        return self

    def predict(self, 
                X):
        """ Predicts the value after the model has been trained.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Test samples
        Returns
        -------
        Predicted value
        """
        # Check correct params
        self.__check_params(X=X)
        return self.intercept_ + np.dot(self.coef_.T, X.T).flatten()

#Create dataset
x = np.random.rand(1000, 10)
y = 2 + 3*x[:, 0].reshape((1000, 1))**2 + np.random.rand(1000, 1)

#Use class LinearRegressionGD
lin_reg = RidgeRegressionGD(random_state=42,
                            plot_loss=False)
lin_reg.fit(X=x,
            y=y,
            learning_rate=0.01,
            C=0.0001,
            max_n_iterations=10000,
            batch_size=128
            )
#Get Predict
prediction = lin_reg.predict(X=x)

#Metrics
print('MAE', mean_absolute_error(y, prediction))
print('MSE', mean_squared_error(y, prediction))
print('RMSE', mean_squared_error(y, prediction, squared=False))
print('MAPE', mean_absolute_percentage_error(y, prediction), '\n')

#Check sklearn model
sk_lin = Ridge(alpha=0.0001,
                max_iter=10000,
                random_state=42)
sk_lin.fit(X=x, 
            y=y)
prediction_sk = sk_lin.predict(X=x)
print('SKLEARN PREDICT')
print('MAE', mean_absolute_error(y, prediction_sk))
print('MSE', mean_squared_error(y, prediction_sk))
print('RMSE', mean_squared_error(y, prediction_sk, squared=False))
print('MAPE', mean_absolute_percentage_error(y, prediction_sk))