import numpy as np
import pandas as pd
import itertools as it
from DecisionTree import DecisionTreeClass, DecisionTreeReg
class PoissonRandomForestRegressor():
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
            y:np.array or pd.DataFrame):
        """function to train the trees

        Args:
            X (np.arrayorpd.DataFrame): _description_
            y (np.arrayorpd.DataFrame): _description_

        Returns:
            _type_: _description_
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
            my_tree = DecisionTreeReg(max_depth=self.__max_depth, min_samples_split=self.__min_samples_split)
            my_tree.fit(X=new_X, y=new_y)
            self.__rf_models.append(my_tree)
        return self


    def predict(self, 
                X:np.array or pd.DataFrame, 
                smooth:int=None)->np.ndarray:
        """function to predict new dataset

        Args:
            X (np.arrayorpd.DataFrame): _description_
            smooth (int, optional): _description_. Defaults to None.

        Raises:
            Exception: _description_

        Returns:
            np.ndarray: _description_
        """
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


class PoissonRandomForestClassifier():
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

    def __bootstrap(self):
            """Function implementation of bootstrap sampling.
            """
            samples = np.random.choice(a = range(self.__x_samples), size = self.__x_samples)
            self.__new_df_indexes = samples

    def __poiss(self):
        """Function implementation of Poisson bootstrap sampling.

        Args:
            data (pd.DataFrame): _description_

        """

        poisson = np.random.poisson(size = self.__x_samples)
        self.__new_df_indexes = []
        for ind, cnt in enumerate(poisson):
            if cnt != 0:
                self.__new_df_indexes += it.repeat(ind, cnt)

    def fit(self, 
            X:np.array, 
            y:np.array):
        """function to train the tree

        Args:
            X (np.arrayorpd.DataFrame): _description_
            y (np.arrayorpd.DataFrame): _description_

        Returns:
            _type_: _description_
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
                X:np.array)->np.ndarray:
        """function to predict new dataset

        Args:
            X (np.arrayorpd.DataFrame): _description_

        Returns:
            np.ndarray: _description_
        """
        if ~isinstance(X, pd.DataFrame):   
            X = pd.DataFrame(X)
        results = [model.predict(X) for model in self.__rf_models]
        result_data = pd.DataFrame(results)
        final_results = np.array([result_data[j].mode()[0] for j in result_data])
        return np.array(final_results)

    def predict_proba(self, 
                        X:np.array or pd.DataFrame)->np.ndarray:
        """function to predict proba new dataset

        Args:
            X (np.arrayorpd.DataFrame): _description_

        Returns:
            np.ndarray: _description_
        """
        if ~isinstance(X, pd.DataFrame):   
            X = pd.DataFrame(X)
        results = [model.predict(X) for model in self.__rf_models]
        result_data = pd.DataFrame(results)
        final_results = np.array([[sum(result_data[j]==0)/len(result_data[j]), sum(result_data[j]==1)/len(result_data[j])] for j in result_data])
        return np.array(final_results)