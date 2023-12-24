import numpy as np


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.split_feature = None
        self.split_threshold = None
        self.left = None
        self.right = None
        self.label = None

    def fit(self, X, y):
        #Check stopping conditions for recursion and set the label for the leaf node.
        if len(np.unique(y)) == 1 or len(X) < self.min_samples_split or self.max_depth == 0:
            self.label = np.round(np.mean(y))
            return
        n_samples, n_features = X.shape
        self.n_features = n_features if self.n_features is None else min(n_features, self.n_features)
        if self.max_depth is not None:  # Add this condition to handle None value of max_depth
            self.max_depth -= 1  # Decrement max_depth by 1
        #Find the best feature and threshold to split the data based on information gain.
        best_gain = -1
        for feature in range(self.n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._calculate_information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    self.split_feature = feature
                    self.split_threshold = threshold
        if best_gain == 0:
            self.label = np.round(np.mean(y))
            return
        #Split the data into left and right nodes and continue the recursion
        left_idxs = X[:, self.split_feature] <= self.split_threshold
        right_idxs = ~left_idxs
        self.left = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                 n_features=self.n_features)
        self.right = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                  n_features=self.n_features)
        self.left.fit(X[left_idxs], y[left_idxs])
        self.right.fit(X[right_idxs], y[right_idxs])


    def predict(self, X):
        if self.label is not None:
            return np.ones(len(X)) * self.label
        left_idxs = X[:, self.split_feature] <= self.split_threshold
        right_idxs = ~left_idxs
        y_pred = np.zeros(len(X))
        y_pred[left_idxs] = self.left.predict(X[left_idxs])
        y_pred[right_idxs] = self.right.predict(X[right_idxs])
        return y_pred

    def _calculate_entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-7))

    def _calculate_information_gain(self, X, y, feature, threshold):
        parent_entropy = self._calculate_entropy(y)
        left_idxs = X[:, feature] <= threshold
        right_idxs = ~left_idxs
        if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
            return 0
        left_entropy = self._calculate_entropy(y[left_idxs])
        right_entropy = self._calculate_entropy(y[right_idxs])
        left_weight = len(y[left_idxs]) / len(y)
        right_weight = len(y[right_idxs]) / len(y)
        return parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)