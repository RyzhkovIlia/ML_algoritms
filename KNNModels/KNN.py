import numpy as np
from scipy.stats import mode

class KNNClassifier(): 
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

    def __find_neighbors(self, x):
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
            d = self.__minkowski(x, query)
            euclidean_distances[ind] = d
        # sort Y_train according to euclidean_distance_array and 
        # store into Y_train_sorted
        inds = euclidean_distances.argsort()
        Y_train_sorted = self.__y_train[inds]
        return Y_train_sorted[:self.n_neighbors]
    
    def __minkowski(self, x, x_train):
        """Function to calculate minkowski distance

        Args:
            x : array-like of shape (n_queries, n_features)
            x_train: array-like of shape (n_queries, n_features)

        Returns:
            ndarray: predict distance
        """
        return np.sum(np.abs(x-x_train))**(1/self.__p)
    
    def fit(self, X, y):
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
            X = np.array(X)
        self.__X_train = X
        self.__y_train = np.array(y)
        return self

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
        # initialize Y_predict
        pred = np.zeros(self.__X_test.shape[0])
        for ind, query in enumerate(self.__X_test):
            # find the K nearest neighbors from current test example
            neighbors = np.zeros(self.n_neighbors)
            neighbors = self.__find_neighbors(query)
            # most frequent class in K neighbors
            pred[ind] = mode(neighbors)[0][0]    
        return pred.flatten()
    
class KNNRegressor(): 
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
            (l2) for p = 2."""
    
    def __init__(self, 
                n_neighbors:int,
                p:int=2):
        self.n_neighbors = n_neighbors
        self.__p = p

    def __find_neighbors(self, x):
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
            d = self.__minkowski(x, query)
            euclidean_distances[ind] = d
        # sort Y_train according to euclidean_distance_array and 
        # store into Y_train_sorted
        inds = euclidean_distances.argsort()
        Y_train_sorted = self.__y_train[inds]
        return Y_train_sorted[:self.n_neighbors]
    
    def __minkowski(self, x, x_train):
        """Function to calculate minkowski distance

        Args:
            x : array-like of shape (n_queries, n_features)
            x_train: array-like of shape (n_queries, n_features)

        Returns:
            ndarray: predict distance
        """
        return np.sum(np.abs(x-x_train))**(1/self.__p)
    
    def fit(self, X, y):
        """Fit the k-nearest neighbors classifier from the training dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : {array-like, sparse matrix} of shape (n_samples,)
            Target values.

        Returns
        -------
        self : KNeighborsRegression
            The fitted k-nearest neighbors regression.
        """
        if isinstance(X, np.ndarray)==False:
            X = np.array(X)
        self.__X_train = X
        self.__y_train = np.array(y)

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
        # initialize Y_predict
        pred = np.zeros(self.__X_test.shape[0])
        for ind, query in enumerate(self.__X_test):
            # find the K nearest neighbors from current test example
            neighbors = np.zeros(self.n_neighbors)
            neighbors = self.__find_neighbors(query)
            # most frequent class in K neighbors
            pred[ind] = np.mean(neighbors) 
        return pred.flatten()