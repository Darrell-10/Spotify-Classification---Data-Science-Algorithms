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
