import numpy as np

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, coeff=None, value=None):
        ''' constructor ''' 
        
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.coeff = coeff
        
        # for leaf node
        self.value = value

class DecisionTreeClass():
    """Classification implementing the Dicision Tree
    max_depth : int, default=2
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node
    """
    def __init__(self, min_samples_split:int=2, max_depth:int=2):
        
        # initialize the root of the tree 
        self.__root = None
        
        # stopping conditions
        self.__min_samples_split = min_samples_split
        self.__max_depth = max_depth

    def __gini_index(self, y):
        ''' function to compute gini index '''
        
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini

    def __information_gain(self, parent, l_child, r_child):
        ''' function to compute information gain '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)

        return self.__gini_index(parent) - (weight_l*self.__gini_index(l_child) + weight_r*self.__gini_index(r_child))

    def __split(self, dataset, feature_index, threshold):
        ''' function to split the data '''
        
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right

    def __get_best_split(self, dataset, num_features):
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
                dataset_left, dataset_right = self.__split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_coeff = self.__information_gain(y, left_y, right_y)
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
    
    def __calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        
        Y = list(Y)
        return max(Y, key=Y.count)

    def __build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree ''' 
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        
        # split until stopping conditions are met
        if num_samples>=self.__min_samples_split and curr_depth<=self.__max_depth:
            # find the best split
            best_split = self.__get_best_split(dataset, num_features)
            # check if information gain is positive
            try:
                if best_split["coeff"]>0:
                    # recur left
                    left_subtree = self.__build_tree(best_split["dataset_left"], curr_depth+1)
                    # recur right
                    right_subtree = self.__build_tree(best_split["dataset_right"], curr_depth+1)
                    # return decision node
                    return Node(best_split["feature_index"], best_split["threshold"], 
                                left_subtree, right_subtree, best_split["coeff"])
            except:
                pass
        
        # compute leaf node
        leaf_value = self.__calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    def __make_prediction(self, x, tree):
        ''' function to predict a single data point '''
        
        if tree.value!=None: 
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.__make_prediction(x, tree.left)
        else:
            return self.__make_prediction(x, tree.right)
        
    def fit(self, X, y):
        ''' function to train the tree '''
        if isinstance(X, np.ndarray)==False:
            X, y = np.array(X), np.array(y)
        y = y.reshape(-1,1)
        dataset = np.concatenate((X, y), axis=1)
        self.__root = self.__build_tree(dataset)
    
    def predict(self, X):
        ''' function to predict new dataset '''
        if isinstance(X, np.ndarray)==False:
            X = np.array(X)
        preditions = [self.__make_prediction(x, self.__root) for x in X]
        return preditions

    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.__root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "Gini =", tree.coeff)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
        
class DecisionTreeReg():
    ''' Regression implementing the Decision Tree
    max_depth : int, default=2
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node
    '''
    
    def __init__(self, min_samples_split:int=2, max_depth:int=2):
        
        # initialize the root of the tree 
        self.__root = None
        
        # stopping conditions
        self.__min_samples_split = min_samples_split
        self.__max_depth = max_depth

    def __variance_reduction(self, parent, l_child, r_child):
        ''' function to compute variance reduction '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        reduction = np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))
        return reduction

    def __split(self, dataset, feature_index, threshold):
        ''' function to split the data '''
        
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    

    def __get_best_split(self, dataset, num_features):
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
                dataset_left, dataset_right = self.__split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_coeff = self.__variance_reduction(y, left_y, right_y)
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
    
    def __calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        
        val = np.mean(Y)
        return val

    def __build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree ''' 
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        
        # split until stopping conditions are met
        if num_samples>=self.__min_samples_split and curr_depth<=self.__max_depth:
            # find the best split
            best_split = self.__get_best_split(dataset, num_features)
            # check if information gain is positive
            try:
                if best_split["coeff"]>0:
                    # recur left
                    left_subtree = self.__build_tree(best_split["dataset_left"], curr_depth+1)
                    # recur right
                    right_subtree = self.__build_tree(best_split["dataset_right"], curr_depth+1)
                    # return decision node
                    return Node(best_split["feature_index"], best_split["threshold"], 
                                left_subtree, right_subtree, best_split["coeff"])
            except:
                pass
        
        # compute leaf node
        leaf_value = self.__calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)

    def __make_prediction(self, x, tree):
        ''' function to predict a single data point '''
        
        if tree.value!=None: 
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.__make_prediction(x, tree.left)
        else:
            return self.__make_prediction(x, tree.right)
        
    def fit(self, X, y):
        ''' function to train the tree '''
        if isinstance(X, np.ndarray)==False:
            X, y = np.array(X), np.array(y)
        y = y.reshape(-1,1)
        dataset = np.concatenate((X, y), axis=1)
        self.__root = self.__build_tree(dataset)
    
    def predict(self, X)->np.array:
        ''' function to predict new dataset '''
        if isinstance(X, np.ndarray)==False:
            X = np.array(X)
        preditions = [self.__make_prediction(x, self.__root) for x in X]
        return np.array(preditions)

    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.__root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "Var =", tree.coeff)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)