import math
import numpy as np
import pandas as pd

class GaussianNaiveBayes:
    def __init__(self):

        """
            Attributes:
                likelihoods: Likelihood of each feature per class
                class_priors: Prior probabilities of classes 
                pred_priors: Prior probabilities of features 
                features: All features of dataset
                                        Likelihood * Class prior probability
        Posterior Probability = -------------------------------------
                                    Predictor prior probability
                    P(B|A) * p(A)
        P(A|B) = ------------------ 
                        P(B)
        """

    def __calc_class_prior(self):
        """Calculate the a priori probability of classes P(y)
        P(A) - Prior Class Probability
        """

        for outcome in np.unique(self.__y_train):
            outcome_count = sum(self.__y_train == outcome)
            # Determine the frequency (a priori probability) and record it in the dictionary
            self.__class_priors[outcome] = outcome_count / self.__train_size

    def __calc_likelihoods(self):

        """Calculate the likelihood table for all functions
        P(B|A) - Likelihood 
        """

        for feature in self.__features:
            for outcome in np.unique(self.__y_train):
                # Find the mean and variance and write them in the likelihood dictionary
                self.__likelihoods[feature][outcome]['mean'] = \
                    self.__X_train[feature][self.__y_train[self.__y_train == outcome].index.values.tolist()].mean()
                self.__likelihoods[feature][outcome]['variance'] = \
                    self.__X_train[feature][self.__y_train[self.__y_train == outcome].index.values.tolist()].var()

    def fit(self, 
            X:pd.DataFrame, 
            y:pd.Series):
        """Fit the training data
        Args:
            X (pd.DataFrame): Pandas dataframe without target feature
            y (pd.Series): Pandas series target feature
        """

        # Plausibility Dictionary
        self.__likelihoods = {}
        # Dictionaries of a priori probability
        self.__class_priors = {}
        # Defining features
        self.__features = list(X.columns)
        self.__X_train = X
        self.__y_train = y
        # Data dimensionality
        self.__train_size = X.shape[0]

        for column in self.__features:
            # We go through each fiche
            # Fill in the plausibility dictionary for each fiche
            self.__likelihoods[column] = {}

            # Updating the dictionaries for each feature
            for outcome in np.unique(self.__y_train):
                self.__likelihoods[column].update({outcome:{}})
                self.__class_priors.update({outcome: 0})

        # Calculate the a priori probability P(A)
        self.__calc_class_prior()
        # Calculate the likelihood parameters P(B|A)
        self.__calc_likelihoods()

        return self
    
    def predict(self, 
                X:pd.DataFrame)->np.array:
        """Predicts the value after the model has been trained.
        Calculates Posterior probability P(c|x) 
        Args:
            X (pd.DataFrame): Pandas dataframe without target feature
        Returns:
            np.array: Predict
        """
        
        results = []
        X = np.array(X)
        for query in X:
            # Going through all the records in the test sample
            probs_outcome = []
            for outcome in np.unique(self.__y_train):
                # We obtain the values of the a priori probability
                prior = self.__class_priors[outcome]
                likelihood = 1

                for column, values in zip(self.__features, query):
                    # We get the average value and the probability
                    mean = self.__likelihoods[column][outcome]['mean']
                    var = self.__likelihoods[column][outcome]['variance']
                    # Find the likelihood value from the Gaussian distribution
                    likelihood *= (1/math.sqrt(2*math.pi*var)) * np.exp(-(values - mean)**2 / (2*var))

                probs_outcome.append(likelihood * prior)

            result = np.argmax(probs_outcome)
            results.append(result)

        return np.array(results)