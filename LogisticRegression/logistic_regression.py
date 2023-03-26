import numpy as np
from scipy.optimize import fmin_tnc
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
                                args=(x, y.flatten()),
                                disp=0)
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
