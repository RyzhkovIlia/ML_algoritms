import numpy as np
import pandas as pd

def _np_check(func):
        """Decorator for check X argument
        """
        def inner(*args, **kwargs):
            for num, arg in enumerate(args):
                if type(arg) == np.ndarray:
                    assert \
                    len(arg) > 0, \
                    f'Argument {num} in function {func} must not be empty.'
            try:
                for key, kwarg in kwargs.items():
                    if type(kwarg) == np.ndarray:
                        assert \
                        len(kwarg) > 0, \
                        f'Argument {key} in function {func} must not be empty.'
            except:
                pass

            return func(*args, **kwargs)
        return inner

class _Node():
    def __init__(self, 
                feature_index:int or None=None, 
                threshold:int or float or None=None, 
                left:np.array or None=None, 
                right:np.array or None=None, 
                coeff:float or None=None, 
                value:float or None=None):
        ''' constructor ''' 
        
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.coeff = coeff
        
        # for leaf node
        self.value = value

class DecisionInfo():
    def __init__(self):
        pass

    @_np_check
    def __gini_index(self, 
                    Y:np.array):
        ''' function to compute gini index '''
        
        class_labels = np.unique(Y)
        gini = 0
        for cls in class_labels:
            p_cls = len(Y[Y == cls]) / len(Y)
            gini += p_cls**2
        return 1 - gini

    @_np_check
    def information_gain(self, 
                            parent:np.array, 
                            l_child:np.array, 
                            r_child:np.array):
        ''' function to compute information gain '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)

        return self.__gini_index(Y=parent) - (weight_l*self.__gini_index(Y=l_child) + weight_r*self.__gini_index(Y=r_child))
    
    @_np_check
    def variance_reduction(self, 
                            parent:np.array, 
                            l_child:np.array, 
                            r_child:np.array):
        ''' function to compute variance reduction '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        reduction = np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))
        return reduction
    
    @_np_check
    def calculate_leaf_value_class(self, 
                                Y:np.array):
        ''' function to compute leaf node '''
        
        Y = list(Y)
        return max(Y, key=Y.count)
    
    @_np_check
    def calculate_leaf_value_reg(self, 
                                Y:np.array):
        ''' function to compute leaf node '''
        
        return np.mean(Y)

class DecisionBuild(DecisionInfo):
    def __init__(self,
                task:str,
                min_samples_split:int,
                max_depth:int):
        super().__init__()
        self.__min_samples_split = min_samples_split
        self.__max_depth = max_depth
        self.__use_func = super().information_gain if task == 'class' else super().variance_reduction
        self.__use_leaf = super().calculate_leaf_value_class if task == 'class' else super().calculate_leaf_value_reg

    @_np_check
    def __split(self, 
                dataset:np.array, 
                feature_index:int, 
                threshold:int or float):
        ''' function to split the data '''
        
        dataset_left = np.array([row for row in dataset if row[feature_index]<threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>=threshold])
        return dataset_left, dataset_right

    @_np_check
    def __get_best_split(self, 
                        dataset:np.array, 
                        num_features:int):
        ''' function to find the best split '''
        
        # dictionary to store the best split
        best_split = {}
        max_coeff = -float("inf")
        
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.__split(dataset=dataset, 
                                                            feature_index=feature_index, 
                                                            threshold=threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_coeff = self.__use_func(parent=y, l_child=left_y, r_child=right_y)
                    # update the best split if needed
                    if curr_coeff>max_coeff:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["coeff"] = curr_coeff
                        max_coeff = curr_coeff
                        
        # return best split
        return best_split

    @_np_check
    def __build_tree(self, 
                    dataset:np.array, 
                    curr_depth:int=0):
        ''' recursive function to build the tree ''' 
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        
        # split until stopping conditions are met
        if num_samples>=self.__min_samples_split and curr_depth<=self.__max_depth:
            # find the best split
            best_split = self.__get_best_split(dataset=dataset, 
                                                num_features=num_features)
            # check if information gain is positive
            try:
                if best_split["coeff"]>0:
                    # recur left
                    left_subtree = self.__build_tree(dataset=best_split["dataset_left"], 
                                                    curr_depth=curr_depth+1)
                    # recur right
                    right_subtree = self.__build_tree(dataset=best_split["dataset_right"], 
                                                    curr_depth=curr_depth+1)
                    # return decision node
                    return _Node(best_split["feature_index"], best_split["threshold"], 
                                left_subtree, right_subtree, best_split["coeff"])
            except:
                pass
        
        # compute leaf node
        leaf_value = self.__use_leaf(Y=Y)
        # return leaf node
        return _Node(value=leaf_value)
    
    def print_tree(self, 
                    tree:_Node or None=None, 
                    indent:str=" "):
        ''' function to print the tree '''
        
        if tree is None:
            tree = self.__root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "Gini =", tree.coeff)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree=tree.left, indent=indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree=tree.right, indent=indent + indent)

    def fit(self, 
            X:np.array or pd.DataFrame or pd.Series,
            y:np.array or pd.Series):
        ''' function to train the tree '''
        if isinstance(X, np.ndarray)==False:
            X, y = np.array(X), np.array(y)
        y = y.reshape(-1,1)
        dataset = np.concatenate((X, y), axis=1)
        self.__root = self.__build_tree(dataset=dataset)
        return self.__root
    
    @_np_check
    def __make_prediction(self, 
                            X:np.array, 
                            tree:_Node):
        ''' function to predict a single data point '''
        if tree.value!=None: 
            return tree.value
        feature_val = X[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.__make_prediction(X=X, tree=tree.left)
        else:
            return self.__make_prediction(X=X, tree=tree.right)
        
    def predict(self, 
                X:np.array or pd.DataFrame or pd.Series,
                tree:_Node)->np.ndarray:
        ''' function to predict new dataset '''
        if isinstance(X, np.ndarray)==False:
            X = np.array(X)
        preditions = [self.__make_prediction(X=x, tree=tree) for x in X]
        return np.array(preditions)

class DecisionTreeClass(DecisionBuild):
    """Classification implementing the Dicision Tree
    max_depth : int, default=2
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node
    """
    def __init__(self, 
                min_samples_split:int=2, 
                max_depth:int=2):
        
        # initialize the root of the tree 
        self.__root = None
        # stopping conditions
        self.__min_samples_split = min_samples_split
        self.__max_depth = max_depth
        self.__check_params()
        super().__init__(task='class', 
                        min_samples_split=min_samples_split,
                        max_depth=max_depth)

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
    
    @_np_check
    def fit(self, 
            X: pd.Series or pd.DataFrame or np.array, 
            y: pd.Series or np.array):
        self.__root = super().fit(X, y)

    @_np_check
    def predict(self, 
                X: np.array or pd.DataFrame or pd.Series):
        return super().predict(X=X, tree=self.__root)

    def print_tree(self, 
                    tree: _Node = None, 
                    indent: str = " "):
        """_summary_

        Args:
            tree (_Node, optional): _description_. Defaults to None.
            indent (str, optional): _description_. Defaults to " ".

        Returns:
            _type_: _description_
        """
        return super().print_tree(tree=tree, 
                                    indent=indent)
    
class DecisionTreeReg(DecisionBuild):
    ''' Regression implementing the Decision Tree
    max_depth : int, default=2
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node
    '''
    
    def __init__(self, 
                min_samples_split:int=2, 
                max_depth:int=2):
        
        # initialize the root of the tree 
        self.__root = None
        # stopping conditions
        self.__min_samples_split = min_samples_split
        self.__max_depth = max_depth
        self.__check_params()
        super().__init__(task='reg', 
                        min_samples_split=min_samples_split,
                        max_depth=max_depth)

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
        
    @_np_check
    def fit(self, 
        X: pd.Series or pd.DataFrame or np.array, 
        y: pd.Series or np.array):
        self.__root = super().fit(X, y)
    
    @_np_check
    def predict(self, 
                X: np.array or pd.DataFrame or pd.Series):
        return super().predict(X=X, tree=self.__root)

    def print_tree(self, 
                    tree: _Node = None, 
                    indent: str = " "):
        """_summary_

        Args:
            tree (_Node, optional): _description_. Defaults to None.
            indent (str, optional): _description_. Defaults to " ".

        Returns:
            _type_: _description_
        """
        return super().print_tree(tree=tree, 
                                    indent=indent)