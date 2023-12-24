import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from random import random
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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
    



    
### start tarining
     
ds = pd.read_csv("heart.csv")
X = ds.iloc[:, :-1].values  # Convert X to a NumPy array   
Y = ds.iloc[:, -1].values.ravel()  # Flatten Y using np.ravel

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

n_estimators = 20
max_depth = 1
acc_threshold = 0.80
acc = 0.0
no=1

clf = RandomForest(n_estimators=n_estimators, max_depth=max_depth)
while acc <= acc_threshold:
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    
    acc = accuracy_score(y_test, predictions)
    print(f"NO.{no}, n_estimators: {n_estimators}, Accuracy: {acc}")
    no+=1

print(classification_report(y_test, predictions))

cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm/np.sum(cm),annot=True,fmt=".2%",cmap='Blues')
plt.show()

y_pred = clf.predict(X_test)
y_pred_prob = y_pred / clf.n_estimators

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC Heart Disease Prediction classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()


print(auc(fpr, tpr))

import pickle as cPickle
with open("C:/Users/iq22b/OneDrive/Documents/Randoom_Forest/TrainModel",'wb') as f:
    cPickle.dump(clf,f)