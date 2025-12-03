"""
Programmer: Jon Larson, 
Class: Cpsc 322-01, Fall 2025
Group Project - Spotify Classification
12/2/25
Description: This file contains
the utility functions used within []
"""

import numpy as np
import mysklearn.myevaluation as myevaluation
from tabulate import tabulate
from collections import Counter
import math

def randomize_in_place(alist, parallel_list=None):
    """
    This function was taken from the 'main.py' file
    within the 'ClassificationFun' folder of 'M4_MLAlgorithmsIntro'
    on GitHub (credit to sluitel2025). It is used to shuffle or 
    randomize the data within 'myevaluation.py' functions such as 
    'train_test_split' and 'kfold_split'.

    Parameters:
        alist: a list that will have its elements shuffled in place
        parallel_list: an optional list that will have the same elements
        shuffled as _alist_.
    
    Notes: _parallel_list_ must be the same length as _alist_ in order to work.
    """
    for i in range(len(alist)):
        # generate a random index to swap this value at i with
        rand_index = np.random.randint(0, len(alist)) # rand int in [0, len(alist))
        # do the swap
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[i]

def cross_val_predict(X, y, classifier_class, k=10, random_state=None, shuffle=True):
    """
    This function performs k-fold cross-validation with a default of 10 folds. It creates
    a new classifier for each of the folds then returns the overall accuracy, error rate,
    true predictions, and predictions.

    Parameters:
        X: the feature data
        y: the target labels to classify
        classifier_class: the class of a classifier wanted to be used to find its accuracy, error rate, true predictions, and overall predictions
        k: the number of folds (defaults to 10)
        random_state: the seed used for reproducibility (defaults to None)
        shuffle: determines if the data is shuffled before splitting (defaults to True)

    Returns:
        tuple: 
            acc: the accuracy of the classifier passed into the 'classifier_class' parameter  
            err: the error rate of the classifier passed into the 'classifier_class' parameter  
            all_true: the true prediction labels made by the classifier from all test folds  
            all_pred: the predicted labels made by the classifier from all test folds  
    """

    folds = myevaluation.kfold_split(X, n_splits=k, random_state=random_state, shuffle=shuffle)
    
    correct = 0
    total = 0

    all_true = []
    all_pred = []

    for train_index, test_index in folds:

        X_train = []
        y_train = []
        for i in train_index:
            X_train.append(X[i])
            y_train.append(y[i])

        X_test = []
        y_test = []
        for i in test_index:
            X_test.append(X[i])
            y_test.append(y[i])

        clf = classifier_class()#instantiate a new classifier instance each fold
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        for i in range(len(y_test)):
            if y_pred[i] == y_test[i]:
                correct += 1
            total += 1

            all_true.append(y_test[i])
            all_pred.append(y_pred[i])

    acc = correct / total
    err = 1 - acc

    return acc, err, all_true, all_pred

def print_confusion_matrix(labels, matrix, title):
    """
    This function prints a confusion matrix as 

    Parameters:
        labels: the column/row labels that correspond to the classified labels
        matrix: the matrix being printed
        title: the title of the matrix being printed
    """
        
    rows = []

    for i, label in enumerate(labels):
        total = sum(matrix[i])
        correct = matrix[i][i]
        recog = (correct / total * 100) if total > 0 else 0
        row = [label] + matrix[i] + [total, f"{recog:.0f}"]
        rows.append(row)

    headers = [labels]

    print(title)
    print(tabulate(rows, headers=headers))

# ----------

def entropy(labels):
    total = len(labels)
    counts = Counter(labels)
    return -sum((count / total) * math.log2(count / total) for count in counts.values() if count != 0)

def information_gain(X, y, attr):
    total_entropy = entropy(y)
    splits = {}
    for i, row in enumerate(X):
        splits.setdefault(row[attr], []).append(i)
    weighted_entropy = 0
    for indices in splits.values():
        subset_y = [y[i] for i in indices]
        weighted_entropy += len(subset_y)/len(y) * entropy(subset_y)
    return total_entropy - weighted_entropy

def majority_class(y):
    """Returns the majority class label. On a tie, chooses the class label
    that comes first alphabetically/in domain order.
    """
    counts = Counter(y)
    # 1. Determine the domain of class labels and sort them (e.g., ["False", "True"] or ["no", "yes"])
    class_labels = sorted(list(set(y)))
    
    best_label = None
    max_count = -1
    
    # 2. Iterate over sorted labels to ensure alphabetical tie-breaking
    for label in class_labels:
        count = counts.get(label, 0)
        if count > max_count:
            max_count = count
            best_label = label
            
    return best_label # Returns the alphabetically first majority label

def build_tree(X, y, attributes, total_count=None):
    if total_count is None:
        total_count = len(y)  # total at root

    if len(set(y)) == 1:
        # Purity leaf
        return ["Leaf", y[0], len(y), total_count]

    if not attributes:
        # No attributes left (Clash)
        maj = majority_class(y)
        return ["Leaf", maj, len(y), total_count]

    # pick best attribute by information gain
    gains = [(attr, information_gain(X, y, attr)) for attr in attributes]
    max_gain = max(gains, key=lambda x: x[1])[1]
    
    # Check for Zero Gain (Clash)
    if max_gain == 0:
        maj = majority_class(y)
        return ["Leaf", maj, len(y), total_count]
        
    best_attrs = [attr for attr, gain in gains if gain == max_gain]
    attr = min(best_attrs)  # tie-breaker: smallest index

    node = ["Attribute", f"att{attr}"]

    # Get unique values and sort them alphabetically
    values = sorted(list(set(row[attr] for row in X)))

    current_node_size = len(y) 

    for val in values:
        indices = [i for i, row in enumerate(X) if row[attr] == val]
        subset_X = [X[i] for i in indices]
        subset_y = [y[i] for i in indices]
        remaining_attrs = [a for a in attributes if a != attr]
        
        # total_count passed to the recursive call is the size of the current node
        if subset_y:
            node.append(["Value", val, build_tree(subset_X, subset_y, remaining_attrs, total_count=current_node_size)])
        else:
            # Handle empty partition by creating a leaf with the parent's majority label and 0 count
            maj_parent = majority_class(y)
            node.append(["Value", val, ["Leaf", maj_parent, 0, current_node_size]])

    return node


def predict_one(x, node):
    if node[0] == "Leaf":
        return node[1]
    _, attr_name, *value_nodes = node
    attr_index = int(attr_name[3:])
    for val_node in value_nodes:
        if val_node[1] == x[attr_index]:
            return predict_one(x, val_node[2])
    # unseen value → majority among leaves at this node
    counts = Counter()
    for val_node in value_nodes:
        leaf = val_node[2]
        if leaf[0] == "Leaf":
            counts[leaf[1]] += leaf[2]
    if counts:
        return counts.most_common(1)[0][0]
    return None

def evaluate_feature_subset(X, y, header, feature_subset, splitter_func, clf_class):
    """
    Runs k-fold CV using split functions and decision tree classifier.

    Parameters:
        X (list of lists): full dataset (same order as header)
        y (list): class labels
        header (list): full header
        feature_subset (list): list of feature names to keep
        splitter_func (function): kfold_split or stratified_kfold_split
        clf_class (class): MyDecisionTreeClassifier class

    Returns:
        float: mean accuracy across folds
    """

    # determine column indices for the chosen features
    col_indices = [header.index(f) for f in feature_subset]

    # Extract only the selected features
    X_sub = [[row[i] for i in col_indices] for row in X]

    # Get fold indices from splitter
    folds = splitter_func(X_sub, y, n_splits=10)

    accuracies = []

    # Loop through folds
    for train_idx, test_idx in folds:
        # Split data
        X_train = [X_sub[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]

        X_test = [X_sub[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]

        # Train tree
        clf = clf_class()
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)

        # Compute accuracy manually
        correct = sum([1 for yt, yp in zip(y_test, y_pred) if yt == yp])
        acc = correct / len(y_test)
        accuracies.append(acc)

    # return mean accuracy
    return sum(accuracies) / len(accuracies)
