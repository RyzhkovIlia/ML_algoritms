from DecisionTree import DecisionTreeReg
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

class GradientBoostingRegression(DecisionTreeReg):
    """Regression implementing the Gradient Bossting
    Parameters
    ----------
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.
        Values must be in the range `(0.0, inf)`.

    max_depth : int, default=3
        Maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. 

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node

        - If int, values must be in the range `[2, inf)`.
    """

    def __init__(self, 
                learning_rate:float=0.1, 
                max_depth:int=3, 
                min_samples_split:int=2):
        self.__max_depth = max_depth
        self.__min_samples_split = min_samples_split
        self.__learning_rate = learning_rate

    def __check_params(self, 
                        verbose:int or None=None, 
                        max_trees:int=100):
        """Check input parameters

        Args:
            verbose (int, optional): Metrics output. Defaults to None.
            max_trees (int, optional):Number of trees. Defaults to 100.
        """
        if isinstance(self.__max_depth, int):
            assert \
            self.__max_depth > 0, \
            'Argument max_depth must be only integer in the range [1, inf)'
        else:
            raise Exception('Argument max_depth must be only integer')
        
        if isinstance(self.__min_samples_split, int):
            assert \
            self.__min_samples_split > 1, \
            'Argument min_samples_split must be only integer in the range [2, inf)'
        else:
            raise Exception('Argument min_samples_split must be only integer')
        
        if (isinstance(self.__learning_rate, int)) | (isinstance(self.__learning_rate, float)):
            assert \
            (self.__learning_rate > 0)&(self.__learning_rate <= 1), \
            'Argument learning_rate must be only integer or float in the range (0, 1]'
        else:
            raise Exception('Argument learning_rate must be only integer or float')
        
        if isinstance(max_trees, int):
            assert \
            max_trees > 0, \
            'Argument max_trees must be only integer in the range [1, inf)'
        else:
            raise Exception('Argument max_trees must be only integer')
        
        if verbose is None:
            pass
        else:
            if isinstance(verbose, int):
                assert \
                verbose > 0, \
                'Argument verbose must be only integer or None in the range [1, inf)'
            else:
                raise Exception('Argument verbose must be only integer or None')

    def _df_check(func):
        """Decorator for check X argument
        """
        def inner(*args, **kwargs):
            key = kwargs['X'] if 'X' in kwargs.keys() else args[1]
            if (isinstance(key, pd.DataFrame)) | (isinstance(key, np.ndarray)):
                    assert \
                    len(key) > 0, \
                    'Argument X must be only pandas DataFrame or numpy ndarray and not empty'
            else:
                raise Exception('Argument X must be only pandas DataFrame or numpy ndarray')
            return func(*args, **kwargs)
        return inner

    @_df_check
    def fit(self, 
            X:np.array or pd.DataFrame, 
            y:np.array or pd.Series, 
            verbose:int or None=None, 
            max_trees:int=100):
        """Fit the gradient boosting model.

        Args:
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                The input samples.

            y : array-like of shape (n_samples,)
                Target values (strings or integers in classification, real numbers
                in regression)
                For classification, labels must correspond to classes.
            verbose (int, optional): Metrics output. Defaults to None.
            max_trees (int, optional):Number of trees. Defaults to 100.

        Returns:
            self : object
            Fitted estimator.
        """

        self.__check_params()
        if (isinstance(y, pd.Series)) | (isinstance(y, np.ndarray)):
            assert \
            len(y) == len(X), \
            'Argument y must be only pandas Series or numpy ndarray and has some X len'
        else:
            raise Exception('Argument y must be only pandas Series or numpy ndarray and has some X len')
        self.__y = y
        self.__trees = []
        y_pred = np.full((y.shape[0], ), np.mean(y))
        for _ in range(max_trees):
            residual = y - y_pred
            tree = DecisionTreeReg(max_depth=self.__max_depth, min_samples_split=self.__min_samples_split)
            tree.fit(X, residual)
            predict_train = tree.predict(X)
            y_pred += self.__learning_rate * predict_train
            self.__trees.append(tree)
            if verbose is not None:
                if _ % verbose == 0:
                    print('Itteration =', _)
                    print('MAE', mean_absolute_error(self.__y,y_pred))
                    print('MSE', mean_squared_error(self.__y,y_pred, squared=False))
                    print('RMSE', mean_absolute_percentage_error(self.__y,y_pred), '\n')
        return self
        
    @_df_check
    def predict(self, 
                X:np.array or pd.DataFrame)->np.ndarray:
        """Predict regression target for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        y_pred = np.mean(self.__y)
        for tree in self.__trees:
            y_pred += self.__learning_rate * tree.predict(X)
        return np.array(y_pred)