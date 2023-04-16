from DecisionTree import DecisionTreeReg
import numpy as np
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
        The minimum number of samples required to split an internal node:

        - If int, values must be in the range `[2, inf)`.
    """
    def __init__(self, 
                learning_rate:float=0.1, 
                max_depth:int=3, 
                min_samples_split:int=2):
        self.__max_depth = max_depth
        self.__min_samples_split = min_samples_split
        self.__learning_rate = learning_rate

    def fit(self, 
            X, 
            y, 
            verbose:int=None, 
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
        
    def predict(self, 
                X)->np.ndarray:
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