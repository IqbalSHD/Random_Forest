import numpy as np
from decisionTree import DecisionTree


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, n_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features

    def fit(self, X, y):
        self.trees = []
        # Get the number of samples and features in the input data X.
        n_samples, n_features = X.shape
        self.n_features = n_features if self.n_features is None else min(n_features, self.n_features)
        # Built n_estimators number of decision trees to create the random forest.
        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                n_features=self.n_features)
            # Perform bootstrapping to create a subset of the data (X_sample, y_sample) and fit the decision tree on this subset.
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample) 
            self.trees.append(tree)



    def predict(self, X):
        y_pred = np.zeros(len(X))
        # Each decision tree in the forest and accumulate their predictions in 
        for tree in self.trees:
            y_pred += tree.predict(X)
        # Average the accumulated predictions by dividing y_pred by the number of trees and rounding them to the nearest integer.
        y_pred /= len(self.trees)
        return np.round(y_pred).astype(int)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]