import numpy as np
import pandas as pd
import itertools as it
from DecisionTree import DecisionTreeClass, DecisionTreeReg
class MyRandomForestRegressor():
    """Regression implementing the Random Forest
    n_estimators : int, default=100
        The number of trees in the forest.

    max_depth : int, default=2
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node

    sample_method : str, default=bootstrap
        Sampling method. Must be only 'bootstrap' or 'poisson'
    """
    def __init__(self,
                n_estimators:int=10,
                max_depth:int=2, 
                min_samples_split:int=2,
                sample_method:str='bootstrap'):
        self.__n_estimators = n_estimators
        self.__max_depth = max_depth
        self.__min_samples_split = min_samples_split
        self.__sample_method = sample_method
        self.__method_list = ['bootstrap', 'poisson']
        self.__check_params()

    def __check_params(self):
        """Check input parameters
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
        
        if isinstance(self.__n_estimators, int):
            assert \
            self.__n_estimators > 0, \
            'Argument n_estimators must be only integer in the range [1, inf)'
        else:
            raise Exception('Argument n_estimators must be only integer')
        
        if isinstance(self.__sample_method, str):
            assert \
            self.__sample_method in self.__method_list, \
            'Argument sample_method must be only string and bootstrap or poisson'
        else:
            raise Exception('Argument sample_method must be only string and bootstrap or poisson')
        
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

    def __bootstrap(self):
        """Function implementation of bootstrap sampling.
        """
        samples = np.random.choice(a = range(self.__x_samples), size = self.__x_samples)
        self.__new_df_indexes = samples

    def __poiss(self):
        """Function implementation of Poisson bootstrap sampling.
        """
        poisson = np.random.poisson(size = self.__x_samples)
        self.__new_df_indexes = []
        for ind, cnt in enumerate(poisson):
            if cnt != 0:
                self.__new_df_indexes += it.repeat(ind, cnt)

    @_df_check
    def fit(self, 
            X:np.array or pd.DataFrame,
            y:np.array or pd.Series):
        """Fit the random forest regression model.

        Args:
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                The input samples.

            y : array-like of shape (n_samples,)
                Target values (strings or integers in classification, real numbers
                in regression)
                For classification, labels must correspond to classes.

        Returns:
            self : object
            Fitted estimator.
        """
        if (isinstance(y, pd.Series)) | (isinstance(y, np.ndarray)):
            assert \
            len(y) == len(X), \
            'Argument y must be only pandas Series or numpy ndarray and has some X len'
        else:
            raise Exception('Argument y must be only pandas Series or numpy ndarray and has some X len')
        
        if ~isinstance(X, pd.DataFrame):   
            X, y = pd.DataFrame(X), pd.Series(y)
        self.__x_samples, self.n_features = X.shape
        self.feature_names_ = np.array(X.columns)
        self.__rf_models = []
        method = self.__bootstrap if self.__sample_method == 'bootstrap' else self.__poiss
        for _ in range(self.__n_estimators):
            method()
            new_X = X.loc[self.__new_df_indexes, :]
            new_y = y.loc[self.__new_df_indexes]
            my_tree = DecisionTreeReg(max_depth=self.__max_depth, min_samples_split=self.__min_samples_split)
            my_tree.fit(X=new_X, y=new_y)
            self.__rf_models.append(my_tree)
        return self

    @_df_check
    def predict(self, 
                X:np.array or pd.DataFrame, 
                smooth:int or None=None)->np.ndarray:
        """Predict regression target for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        smooth : {integer or None}. Defaults to 100

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        assert \
        (isinstance(smooth, int)) | (smooth is None), \
        'Argument smooth must be only integer or None'
        
        if ~isinstance(X, pd.DataFrame):   
            X = pd.DataFrame(X)
        if smooth != None and (smooth>=self.__n_estimators//2 or smooth < 1):
            raise Exception(f"Smooth must be <= {self.__n_estimators//2} and > 0")

        results = [model.predict(X) for model in self.__rf_models]
        if smooth != None:
            self.__smooth = smooth
            results = np.sort(np.asarray(results), axis=0)[self.__smooth:-self.__smooth].sum(axis=0)
        else: 
            self.__smooth = 0
            results = np.asarray(results).sum(axis=0)
        final_results = np.array([result/(self.__n_estimators-(self.__smooth *2)) for result in results])
        return np.array(final_results)

class MyRandomForestClassifier():
    """Classification implementing the Random Forest
    n_estimators : int, default=100
        The number of trees in the forest.

    max_depth : int, default=2
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node

    sample_method : str, default=bootstrap
        Sampling method. Must be only 'bootstrap' or 'poisson'
    """
    def __init__(self,
                n_estimators:int=10,
                max_depth:int=2, 
                min_samples_split:int=2,
                sample_method:str='bootstrap'):
        self.__n_estimators = n_estimators
        self.__max_depth = max_depth
        self.__min_samples_split = min_samples_split
        self.__sample_method = sample_method
        self.__method_list = ['bootstrap', 'poisson']
        self.__check_params()

    def __check_params(self):
        """Check input parameters
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
        
        if isinstance(self.__n_estimators, int):
            assert \
            self.__n_estimators > 0, \
            'Argument n_estimators must be only integer in the range [1, inf)'
        else:
            raise Exception('Argument n_estimators must be only integer')
        
        if isinstance(self.__sample_method, str):
            assert \
            self.__sample_method in self.__method_list, \
            'Argument sample_method must be only string and bootstrap or poisson'
        else:
            raise Exception('Argument sample_method must be only string and bootstrap or poisson')

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

    def __bootstrap(self):
            """Function implementation of bootstrap sampling.
            """
            samples = np.random.choice(a = range(self.__x_samples), size = self.__x_samples)
            self.__new_df_indexes = samples

    def __poiss(self):
        """Function implementation of Poisson bootstrap sampling.
        """

        poisson = np.random.poisson(size = self.__x_samples)
        self.__new_df_indexes = []
        for ind, cnt in enumerate(poisson):
            if cnt != 0:
                self.__new_df_indexes += it.repeat(ind, cnt)

    def fit(self, 
            X:np.array or pd.DataFrame, 
            y:np.array or pd.Series):
        """Fit the random forest regression model.

        Args:
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                The input samples.

            y : array-like of shape (n_samples,)
                Target values (strings or integers in classification, real numbers
                in regression)
                For classification, labels must correspond to classes.

        Returns:
            self : object
            Fitted estimator.
        """
        if ~isinstance(X, pd.DataFrame):   
            X, y = pd.DataFrame(X), pd.Series(np.array(y))
        self.__x_samples, self.n_features = X.shape
        self.feature_names_ = np.array(X.columns)
        self.__rf_models = []
        method = self.__bootstrap if self.__sample_method == 'bootstrap' else self.__poiss
        for _ in range(self.__n_estimators):
            method()
            new_X = X.loc[self.__new_df_indexes, :]
            new_y = y.loc[self.__new_df_indexes]
            my_tree = DecisionTreeClass(max_depth=self.__max_depth, min_samples_split=self.__min_samples_split)
            my_tree.fit(X=new_X, y=new_y)
            self.__rf_models.append(my_tree)
        return self
    
    def predict(self, 
                X:np.array or pd.DataFrame)->np.ndarray:
        """Predict classification target for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        if ~isinstance(X, pd.DataFrame):   
            X = pd.DataFrame(X)
        results = [model.predict(X) for model in self.__rf_models]
        result_data = pd.DataFrame(results)
        final_results = np.array([result_data[j].mode()[0] for j in result_data])
        return np.array(final_results)

    def predict_proba(self, 
                        X:np.array or pd.DataFrame)->np.ndarray:
        """Predict probalistic classification target for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        if ~isinstance(X, pd.DataFrame):   
            X = pd.DataFrame(X)
        results = [model.predict(X) for model in self.__rf_models]
        result_data = pd.DataFrame(results)
        final_results = np.array([[sum(result_data[j]==0)/len(result_data[j]), sum(result_data[j]==1)/len(result_data[j])] for j in result_data])
        return np.array(final_results)