import numpy as np

class LinearRegressionGD:
    """Linear Regression Using Gradient Descent.
        Also Ridge, Lasso and Elastic Net regression
    Parameters
    ----------
    eta : str
        method: Must be linear, lasso, ridge or elastic
    Attributes
    ----------
    coef_ : weights/ after fitting the model
    intercept_ : free member of regression
    """

    def __init__(self,
                method:str='linear'):
        method_list = ['linear', 'lasso', 'ridge', 'elastic']
        assert \
            method.lower() in method_list, \
            f'This method not found. Please give one of this method {method_list}. Receive method = {method.lower()}'
        self.__method = method

    def __check_params(self):
        """Check input params
        """
        assert\
            (isinstance(self.__learning_rate, float))|(isinstance(self.__learning_rate, int))&(self.__learning_rate>0)&(self.__learning_rate<=1),\
            f'Learning_rate must be only in interval (0, 1]. Receive {self.__learning_rate}.'
        assert\
            isinstance(self.__n_iterations, int),\
            f'N_iterations must be only integer. Receive {type(self.__n_iterations)}.'
        assert\
            isinstance(self.__l2_penalty, float)|isinstance(self.__l2_penalty, int),\
            f'L2_penalty must be only integer or float. Receive {type(self.__l2_penalty)}.'
        assert\
            isinstance(self.__l1_penalty, float)|isinstance(self.__l1_penalty, int),\
            f'L1_penalty must be only integer or float. Receive {type(self.__l1_penalty)}.'
        assert\
            (self.__X.shape[0]>0)&(self.__X.shape[1]>0),\
            f'X must not be empty.'
        assert\
            (self.__y.shape[0]==self.__X.shape[0]),\
            f'Y shape must be equal X shape.'
        if self.__method == 'elastic':
            assert\
            (self.__l1_penalty is not None)&(self.__l2_penalty is not None)

    def __calculate_gradient(self):
        if self.__method.lower() == 'ridge':
            # If we have a ridge regression we find the gradient
            self.__dW = ((-2*np.dot(self.__X.T, self.__residuals))+\
                            (2*self.__l2_penalty*self.coef_))/self.__m
        elif self.__method.lower() == 'linear':
            # If the usual linear regression we find the gradient
            self.__dW = (-2*np.dot(self.__X.T, self.__residuals))/self.__m
        elif self.__method.lower() == 'lasso':
            # If we have a lasso regression we find the gradient
            # Create a gradient matrix
            self.__dW = np.zeros(self.n_features_in_)
            # Going over each weight
            for j in range(self.n_features_in_) :
                if self.coef_[j]>0 :
                    self.__dW[j]=((-2*np.dot(self.__X.T, self.__residuals))+\
                                    (self.__l1_penalty*self.coef_[j]))/self.__m
                else :
                    self.__dW[j]=((-2*np.dot(self.__X.T, self.__residuals))-\
                                    (self.__l1_penalty*self.coef_[j]))/self.__m
        else:
            # If we have elastic regression find the gradient
            # Create a gradient matrix
            self.__dW = np.zeros(self.n_features_in_)
            # Going over each weight
            for j in range(self.n_features_in_) :
                if self.coef_[j]>0 :
                    self.__dW[j]=((-2*np.dot(self.__X.T, self.__residuals))+\
                                    (self.__l1_penalty*self.coef_[j])+\
                                    (2*self.__l2_penalty*self.coef_))/self.__m
                else :
                    self.__dW[j]=((-2*np.dot(self.__X.T, self.__residuals))-\
                                    (self.__l1_penalty*self.coef_[j])+\
                                    (2*self.__l2_penalty*self.coef_))/self.__m
        # Find the gradient for the free term b
        self.__db = - 2 * np.sum(self.__residuals) / self.__m 
    
    def __update_weights(self):
        
        y_pred = self.predict(self.__X)
        # Deviation from the true value
        self.__residuals = self.__y - y_pred
        # Calculate gradients  
        self.__calculate_gradient()
        # Update the weights with the gradient found from the production MSE
        self.coef_ -= self.__learning_rate*self.__dW
        # Update free member b
        self.intercept_ -= self.__learning_rate * self.__db

    def fit(self, 
            x, 
            y,
            learning_rate:float=None,
            n_iterations:int=None,
            l2_penalty:float=None,
            l1_penalty:float=None):
        """Fit the training data
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training samples
        y : array-like, shape = [n_samples, n_target_values]
            Target values
        learning_rate: float, learning rate coeff
        n_iterations: int, count of inerations
        l2_penalty: float, penalty coeff if method = ridge or elastic
        l1_penalty: float, penalty coeff if method = lasso or elastic
        Returns
        -------
        self : object
        """
        self.__X = x
        self.__y = y
        self.__learning_rate = learning_rate if learning_rate is not None else 0.001
        self.__n_iterations = n_iterations if n_iterations is not None else 1000
        self.__l2_penalty = l2_penalty if l2_penalty is not None and self.__method in ['ridge', 'elastic'] else None
        self.__l1_penalty = l1_penalty if l1_penalty is not None and self.__method in ['lasso', 'elastic'] else None

        # Check correct params
        self.__check_params()
        # Dimensionality of features
        self.n_features_in_ = self.__X.shape[1]
        # Create a matrix for the weights of each attribute
        self.coef_ = np.zeros((self.n_features_in_, 1))
        # Number of training examples
        self.__m = self.__X.shape[0]
        # Free member b
        self.intercept_ = 0
        for _ in range(self.__n_iterations):
            # Updating the weights
            self.__update_weights() 
        return self

    def predict(self, 
                x):
        """ Predicts the value after the model has been trained.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Test samples
        Returns
        -------
        Predicted value
        """
        return self.intercept_ + np.dot(x, self.coef_)