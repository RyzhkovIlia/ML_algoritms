import numpy as np
import pandas as pd
from scipy.optimize import fmin_tnc
# import matplotlib.pyplot as plt

# marks_df = pd.read_csv("marks.txt", header=None)
# # X = feature values, all the columns except the last column
# X = marks_df.iloc[:, :-1]

# # y = target values, last column of the data frame
# y = marks_df.iloc[:, -1]

class LogisticRegressionGD:
    def __init__(self):
        """Logistic Regression Using Gradient Descent.
        """
        pass
    
    def __sigmoid(self, 
                x):
        """Activation function - sigmoid

        Args:
            x (array-like): Training samples

        Returns:
            _type_: probability after passing through sigmoid
        """
        return 1 / (1 + np.exp(-x))
    
    def __net_input(self, 
                    theta,
                    x):
        """Scalar product

        Args:
            x (array-like): Training samples

        Returns:
            _type_: _description_
        """
        # Computes the weighted sum of inputs
        return np.dot(x, theta.T)

    def __probability(self,
                    theta,
                    x):
        """Probability predict

        Args:
            x (array-like): Training samples

        Returns:
            _type_: Probability
        """
        # Returns the probability after passing through sigmoid
        return self.__sigmoid(self.__net_input(theta, x))
    
    def __cost_function(self, 
                        theta,
                        x, 
                        y):
        """_summary_

        Args:
            x (array-like): Training samples
            y (array-like): Target values

        Returns:
            _type_: _description_
        """
        # Computes the cost function for all the training samples
        m = x.shape[0]
        total_cost = -(1/m)*np.sum(
            y * np.log(self.__probability(theta, x))+(1-y)*np.log(1-self.__probability(theta, x)))
        return total_cost

    def __gradient(self, 
                    theta,
                    x, 
                    y):
        """Computes the gradient of the cost function at the point theta

        Args:
            x (array-like): Training samples
            y (array-like): Target values

        Returns:
            _type_: Gradient
        """
        m = x.shape[0]
        return (1/m)*np.dot(x.T, self.__sigmoid(self.__net_input(theta, x))-y)
    
    def fit(self, 
            x, 
            y, 
            theta:np.ndarray):
        """Fit the training data
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training samples
        y : array-like, shape = [n_samples, n_target_values]
            Target values
        theta: np.ndarray, Matrix for the weights of each attribute
        Returns
        -------
        self : object
        """

        self.opt_weights = fmin_tnc(func=self.__cost_function, 
                                x0=theta, 
                                fprime=self.__gradient,
                                args=(x, y.flatten()))
        return self
    
    def predict(self, x):
        """ Predicts the value after the model has been trained.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Test samples
        Returns
        -------
        Predicted value
        """
        theta = self.opt_weights[0][:, np.newaxis]
        return self.__probability(theta.T, x).flatten()
    
# X = np.c_[np.ones((X.shape[0], 1)), X]
# y = y[:, np.newaxis]
# # Create a matrix for the weights of each attribute
# theta = np.zeros((X.shape[1], 1))
# log_reg = LogisticRegressionGD()
# log_reg.fit(X, y, theta)
# # pred = log_reg.predict(X)
# # pred = (pred>0.5).astype(int)
# # df = pd.DataFrame({"true":y.flatten(), 'pred':pred})
# x_values = [np.min(X[:, 1] - 5), np.max(X[:, 2] + 5)]
# params = log_reg.opt_weights[0]
# y_values = - (params[0] + np.dot(params[1], x_values)) / params[2]

# admitted = marks_df.loc[y == 1]
# not_admitted = marks_df.loc[y == 0]
# plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
# plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
# plt.plot(x_values, y_values, label='Decision Boundary')
# plt.xlabel('Marks in 1st Exam')
# plt.ylabel('Marks in 2nd Exam')
# plt.legend()
# plt.show()
