import numpy as np
class LinearRegressionGD:
    """Linear Regression Using Gradient Descent.
        Also Ridge, Lasso and ElasticNet regression
    Parameters
    ----------
    eta : str
        method: Must be linear, lasso, ridge or elasticnet
    Attributes
    ----------
    coef_ : weights/ after fitting the model
    intercept_ : free member of regression
    """

    def __init__(self,
                penalty:str=None):
        method_list = ['l1', 'l2', 'elasticnet', None]
        assert \
            (penalty is None)|(penalty in method_list), \
            f'This method not found. Please give one of this method {method_list}. Receive method = {penalty}'
        self.__penalty = penalty

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
            isinstance(self.__C, float)|isinstance(self.__C, int),\
            f'C must be only integer or float. Receive {type(self.__C)}.'
        assert\
            (self.__X.shape[0]>0)&(self.__X.shape[1]>0),\
            f'X must not be empty.'
        assert\
            (self.__y.shape[0]==self.__X.shape[0]),\
            f'Y shape must be equal X shape.'


    def __calculate_gradient(self):
        """Function for find out weight and b gradients
        """
        if self.__penalty == None:
            # If the usual linear regression we find the gradient
            self.__dW = (-2*np.sum(np.dot(self.__X.T, self.__residuals)))/self.__m

        elif self.__penalty == 'l2':
            # If we have a ridge regression we find the gradient
            self.__dW = ((-2*np.sum(np.dot(self.__X.T, self.__residuals)))+\
                            (2*self.__C*np.sum(self.coef_)))/self.__m
            
        elif self.__penalty == 'l1':
            # If we have a lasso regression we find the gradient
            # Create a gradient matrix
            self.__dW = np.zeros(self.n_features_in_)
            # Going over each weight
            for j in range(self.n_features_in_) :
                if self.coef_[j]>0 :
                    # print(self.__X)
                    self.__dW[j]=(((-2*np.sum(np.dot(self.__X[:, j].T, self.__residuals)))+\
                                    (self.__C*np.sum(self.coef_[j][0])))/self.__m)
                else :
                    self.__dW[j]=(((-2*np.sum(np.dot(self.__X[:, j].T, self.__residuals)))-\
                                    (self.__C*np.sum(self.coef_[j][0])))/self.__m)
                    
        else:
            # If we have elastic regression find the gradient
            # Create a gradient matrix
            self.__dW = np.zeros(self.n_features_in_)
            # Going over each weight
            for j in range(self.n_features_in_) :
                if self.coef_[j]>0 :
                    self.__dW[j]=((-2*np.sum(np.dot(self.__X[:, j].T, self.__residuals)))+\
                                    (self.__C*np.sum(self.coef_[j]))+\
                                    (2*self.__C*np.sum(self.coef_)))/self.__m
                else :
                    self.__dW[j]=((-2*np.sum(np.dot(self.__X[:, j].T, self.__residuals)))-\
                                    (self.__C*np.sum(self.coef_[j]))+\
                                    (2*self.__C*np.sum(self.coef_)))/self.__m
        # Find the gradient for the free term b
        self.__db = (-2*np.sum(self.__residuals))/self.__m 
    
    def __update_weights(self):
        """Update weights and b
        """
        y_pred = self.predict(self.__X)
        # Deviation from the true value
        self.__residuals = self.__y-y_pred
        if np.sum(self.__residuals) < -10e3:
            self.__flag = False
        # Calculate gradients  
        self.__calculate_gradient()
        # Update the weights with the gradient found from the production MSE
        self.coef_ -= (self.__learning_rate*self.__dW).reshape((self.n_features_in_,1))
        # Update free member b
        self.intercept_ -= self.__learning_rate * self.__db

    def fit(self, 
            x, 
            y,
            learning_rate:float=0.001,
            C:float=1.0,
            n_iterations:int=1000):
        """Fit the training data
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training samples
        y : array-like, shape = [n_samples, n_target_values]
            Target values
        learning_rate: float, learning rate coeff
        C: float , Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
        n_iterations: int, count of inerations
        Returns
        -------
        self : object
        """
        self.__X = x
        self.__y = y
        self.__learning_rate = learning_rate
        self.__n_iterations = n_iterations
        self.__C = C
        self.__flag = True

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
            if self.__flag:
            # Updating the weights
                self.__update_weights()
            else:
                break
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
