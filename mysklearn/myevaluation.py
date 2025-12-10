"""
Programmer: Jon Larson
Class: Cpsc 322-01, Fall 2025
Group Project - Spotify Classification
12/9/25
Description: This file contains useful 
functions for classification tasks.
"""

from mysklearn import myutils

import numpy as np

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """

    if(len(X) != len(y)):#check if the size of X and y are the same
        raise ValueError("X and y must be the same length to call `train_test_split()`")
    
    n = len(X)#length of X (n_samples)

    if isinstance(test_size, int):#check if test_size is passed in as an int
        if not 0 < test_size < n:
            raise ValueError("test_size is an int, the value must be between 0 and the size n_samples - 1")
        n_test = test_size
    elif isinstance(test_size, float):#check if test_size is passed in as a float
        if not 0 < test_size < 1:
            raise ValueError("test_size is a float, the value must be between 0 and 1.")
        n_test = int(np.ceil(n * test_size))
    else:#type error if test_size is not an int or a float
        raise TypeError("test_size must be an int or a float.")
    
    indexes = list(range(n))#list of indexes to be used for shuffling

    if random_state is not None:#seed with random_state if needed
        np.random.seed(random_state)
    
    if shuffle:#shuffle indexes if needed
        myutils.randomize_in_place(indexes)

    X_shuffled = [X[i] for i in indexes]#x shuffle
    y_shuffled = [y[i] for i in indexes]#y shuffle

    #split
    split_index = len(X_shuffled) - n_test

    X_test = X_shuffled[split_index:]
    y_test = y_shuffled[split_index:]

    X_train = X_shuffled[:split_index]
    y_train = y_shuffled[:split_index]

    return X_train, X_test, y_train, y_test

def stratified_train_test_split(X, y, test_size=0.33, random_state=None):
    """
    Split dataset into train and test sets, preserving class distribution (stratified).
    
    Args:
        X (list of list): Features
        y (list): Labels
        test_size (float or int): Proportion or number of instances in test set
        random_state (int): Seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    if len(X) != len(y):
        raise ValueError("X and y must be the same length")
    
    n = len(X)
    
    #number of test samples
    if isinstance(test_size, float):
        n_test_total = int(np.ceil(n * test_size))
    elif isinstance(test_size, int):
        n_test_total = test_size
    else:
        raise TypeError("test_size must be float or int")
    
    rng = np.random.default_rng(random_state)
    
    label_to_indices = {}
    for i, label in enumerate(y):
        label_to_indices.setdefault(label, []).append(i)
    
    X_train, X_test, y_train, y_test = [], [], [], []
    
    for label, indices in label_to_indices.items():
        n_label = len(indices)
        n_test_label = int(np.ceil(n_label * test_size)) if isinstance(test_size, float) else int(np.ceil(n_test_total * n_label / n))
        
        indices_shuffled = list(indices)
        rng.shuffle(indices_shuffled)
        
        test_indices = indices_shuffled[:n_test_label]
        train_indices = indices_shuffled[n_test_label:]
        
        X_test.extend([X[i] for i in test_indices])
        y_test.extend([y[i] for i in test_indices])
        X_train.extend([X[i] for i in train_indices])
        y_train.extend([y[i] for i in train_indices])
    
    return X_train, X_test, y_train, y_test

def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """

    n = len(X)#length of X (n_samples)

    if n_splits <= 1:#check if n_splits is less than or equal to 1
        raise ValueError("n_splits must be at least 2")
    if n_splits > n:#check if n_splits is greater than the number of samples
        raise ValueError("n_splits must be less than n_samples")

    indexes = list(range(n))#list of indexes to be used for shuffling

    if random_state is not None:#seed with random_state if needed
        np.random.seed(random_state)
    
    if shuffle:#shuffle indexes if needed
        myutils.randomize_in_place(indexes)

    #fold size
    base_size = n // n_splits
    remainder = n % n_splits

    fold_sizes = []
    for i in range(n_splits):
        if i < remainder:
            fold_sizes.append(base_size + 1)
        else:
            fold_sizes.append(base_size)

    #create folds
    folds = []
    start = 0

    for size in fold_sizes:
        test_index = indexes[start:start + size]
        train_index = indexes[:start] + indexes[start + size:]
        folds.append((train_index, test_index))
        start += size

    return folds

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """

    if n_samples is None:
        n = len(X)
    else:
        n = n_samples

    if random_state is not None:#seed with random_state if needed
        np.random.seed(random_state)

    sample_indices = [np.random.randint(0, len(X)-1) for _ in range(n)]

    oob_indices = [i for i in range(len(X)) if i not in sample_indices]

    X_sample = [X[i] for i in sample_indices]
    X_out_of_bag = [X[i] for i in oob_indices]

    if y is not None:
        y_sample = [y[i] for i in sample_indices]
        y_out_of_bag = [y[i] for i in oob_indices]
    else:
        y_sample = None
        y_out_of_bag = None

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """

    n_labels = len(labels)

    matrix = []
    for i in range(n_labels):
        row = []
        for j in range(n_labels):
            row.append(0)
        matrix.append(row)
    
    label_to_index = {}#map 
    for i in range(len(labels)):
        label_to_index[labels[i]] = i
    
    for i in range(len(y_true)):#count
        true_val = y_true[i]
        pred_val = y_pred[i]
        j = label_to_index[true_val]
        k = label_to_index[pred_val]
        matrix[j][k] += 1

    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """

    correct_count = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct_count += 1
    
    if normalize:
        return correct_count / len(y_true)
    else:
        return correct_count

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """

    if labels is None:
        labels = []
        for y in y_true:
            if y not in labels:
                labels.append(y)

    if pos_label is None:
        pos_label = labels[0]

    tp = 0
    fp = 0

    for i in range(len(y_true)):
        yt = y_true[i]
        yp = y_pred[i]

        if yp == pos_label:
            if yt == pos_label:
                tp += 1
            else:
                fp += 1

    precision = 0.0

    if tp + fp == 0:
        return precision
    
    precision = tp / (tp + fp)

    return precision

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """

    if labels is None:
        labels = []
        for y in y_true:
            if y not in labels:
                labels.append(y)

    if pos_label is None:
        pos_label = labels[0]

    tp = 0
    fn = 0

    for i in range(len(y_true)):
        yt = y_true[i]
        yp = y_pred[i]

        if yt == pos_label:
            if yp == pos_label:
                tp += 1
            else:
                fn += 1

    recall = 0.0

    if tp + fn == 0:
        return recall
    
    recall = tp / (tp + fn)

    return recall

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    
    if labels is None:
        labels = []
        for y in y_true:
            if y not in labels:
                labels.append(y)

    if pos_label is None:
        pos_label = labels[0]

    precision = binary_precision_score(y_true, y_pred, labels=labels, pos_label=pos_label)

    recall = binary_recall_score(y_true, y_pred, labels=labels, pos_label=pos_label)

    f1 = 0.0

    if precision + recall == 0:
        return f1
    
    f1 = (2 * (precision * recall)) / (precision + recall)

    return f1
