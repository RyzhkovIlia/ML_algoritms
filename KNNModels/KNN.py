import numpy as np
import pandas as pd
from scipy.stats import mode

def _df_np_check(func):
    """Decorator for check X argument
    """
    def inner(*args, **kwargs):
        key = kwargs['X'] if 'X' in kwargs.keys() else args[1]
        if (isinstance(key, pd.DataFrame)) | (isinstance(key, np.ndarray)):
            assert \
            len(key) > 0, \
            'Argument X must be only pandas DataFrame and not empty'
        else:
            raise Exception('Argument X must be only pandas DataFrame')
        return func(*args, **kwargs)
    return inner

class _KNNTools():
    def __init__(self,
                n_neighbors:int,
                p:int,
                X_train:np.array or None = None,
                y_train:np.array or None = None):
        self.__n_neighbors = n_neighbors
        self.__p = p
        self.__X_train = X_train
        self.__y_train = y_train

    def _find_neighbors(self, x):
        """Function to find the K nearest neighbors to current test example 

        Parameters
        ----------
        x : array-like of shape (n_queries, n_features)
            Test samples.

        Returns:
            ndarray: k nearest neighbors
        """
        # calculate all the minkowski distances between current 
        euclidean_distances = np.zeros(self.__X_train.shape[0])
        for ind, query in enumerate(self.__X_train):
            d = self._minkowski(x, query)
            euclidean_distances[ind] = d
        # sort Y_train according to euclidean_distance_array and 
        # store into Y_train_sorted
        inds = euclidean_distances.argsort()
        Y_train_sorted = self.__y_train[inds]
        if self.__n_neighbors > len(Y_train_sorted):
            self.__n_neighbors = len(Y_train_sorted)-1
        return Y_train_sorted[:self.__n_neighbors]
    
    def _minkowski(self, x, x_train):
        """Function to calculate minkowski distance

        Args:
            x : array-like of shape (n_queries, n_features)
            x_train: array-like of shape (n_queries, n_features)

        Returns:
            ndarray: predict distance
        """
        return np.sum(np.abs(x-x_train))**(1/self.__p)
    
    def _fit(self, 
            X:pd.DataFrame or np.array, 
            y:pd.Series or np.array):
        """Fit the k-nearest neighbors classifier from the training dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : {array-like, sparse matrix} of shape (n_samples,)
            Target values.

        Returns
        -------
        self : KNeighborsClassifier
            The fitted k-nearest neighbors classifier.
        """
        if isinstance(X, np.ndarray)==False:
            self.n_features = X.shape[1]
            self.feature_names_ = np.array(X.columns)
            X = np.array(X)
        if (isinstance(y, pd.Series)) | (isinstance(y, np.ndarray)):
            assert \
            len(y) == len(X), \
            'Argument y must be only pandas Series and has some X len'
        else:
            raise Exception('Argument y must be only pandas Series and has some X len')
        return X, y

    def _check_params(self):
        """Check input parameters
        """
        if isinstance(self.__n_neighbors, int):
            assert \
            self.__n_neighbors > 0, \
            'Argument n_neighbors must be only integer in the range [1, inf)'
        else:
            raise Exception('Argument n_neighbors must be only integer')
        
        if isinstance(self.__p, int):
            assert \
            self.__p in [1, 2], \
            'Argument p must be only integer in the range [1, 2]'
        else:
            raise Exception('Argument p must be only integer')
        
class KNNClassifier(_KNNTools): 
    """Classifier implementing the k-nearest neighbors vote.
    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for :meth:`kneighbors` queries.

    p : int, default=2
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2.
    """
    def __init__(self, 
                n_neighbors:int,
                p:int=2):
                
        self.n_neighbors = n_neighbors
        self.__p = p
        super().__init__(n_neighbors=self.n_neighbors,
                        p=self.__p)
        super()._check_params()
        
    @_df_np_check
    def fit(self, 
            X:pd.DataFrame or np.array, 
            y:pd.Series or np.array):
        """Fit the k-nearest neighbors classifier from the training dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : {array-like, sparse matrix} of shape (n_samples,)
            Target values.

        Returns
        -------
        self : KNeighborsClassifier
            The fitted k-nearest neighbors classifier.
        """
        self.__X_train, self.__y_train = super()._fit(X=X, 
                                                        y=y)
        return self

    @_df_np_check
    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs)
            Class labels for each data sample.
        """
        if isinstance(X, np.ndarray)==False:
            X = np.array(X)
        self.__X_test = X
        super().__init__(X_train = self.__X_train,
                        y_train = self.__y_train,
                        n_neighbors=self.n_neighbors,
                        p=self.__p)
        # initialize Y_predict
        pred = np.zeros(self.__X_test.shape[0])
        for ind, query in enumerate(self.__X_test):
            # find the K nearest neighbors from current test example
            neighbors = np.zeros(self.n_neighbors)
            neighbors = super()._find_neighbors(query)
            # most frequent class in K neighbors
            pred[ind] = mode(neighbors, keepdims = True)[0][0]    
        return pred.flatten()
    
class KNNRegressor(_KNNTools): 
    """Regression based on k-nearest neighbors.

        The target is predicted by local interpolation of the targets
        associated of the nearest neighbors in the training set.

        Parameters
        ----------
        n_neighbors : int, default=5
            Number of neighbors to use by default for :meth:`kneighbors` queries.

        p : int, default=2
            Power parameter for the Minkowski metric. When p = 1, this is
            equivalent to using manhattan_distance (l1), and euclidean_distance
            (l2) for p = 2. Default = 2"""
    
    def __init__(self, 
                n_neighbors:int,
                p:int=2):
        self.n_neighbors = n_neighbors
        self.__p = p
        super().__init__(n_neighbors=self.n_neighbors,
                        p=self.__p)
        super()._check_params()
    
    @_df_np_check
    def fit(self, 
            X:pd.DataFrame or np.array, 
            y:pd.Series or np.array):
        """Fit the k-nearest neighbors regression from the training dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : {array-like, sparse matrix} of shape (n_samples,)
            Target values.

        Returns
        -------
        self : KNeighborsClassifier
            The fitted k-nearest neighbors regression.
        """
        self.__X_train, self.__y_train = super()._fit(X=X, 
                                                        y=y)
        return self

    @_df_np_check
    def predict(self, X):
        """Predict the target for the provided data.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_queries,)
            Target values.
        """
        if isinstance(X, np.ndarray)==False:
            X = np.array(X)
        self.__X_test = X
        super().__init__(X_train = self.__X_train,
                        y_train = self.__y_train,
                        n_neighbors=self.n_neighbors,
                        p=self.__p)
        # initialize Y_predict
        pred = np.zeros(self.__X_test.shape[0])
        for ind, query in enumerate(self.__X_test):
            # find the K nearest neighbors from current test example
            neighbors = np.zeros(self.n_neighbors)
            neighbors = super()._find_neighbors(query)
            # most frequent class in K neighbors
            pred[ind] = np.mean(neighbors) 
        return pred.flatten()