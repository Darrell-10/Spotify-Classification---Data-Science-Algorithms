"""
Programmer: Jon Larson, Darrell Cenido
Class: Cpsc 322-01, Fall 2025
Group Project - Spotify Classification
12/9/25
Description: This file contains
the classes used to fit and predict
labels for classification
"""

from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
from mysklearn import myutils
from mysklearn import myevaluation
import numpy as np
from collections import Counter

class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        if self.regressor is None:
            self.regressor = MySimpleLinearRegressor()
        
        self.regressor.fit(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        preds = self.regressor.predict(X_test)

        y_predicted = [self.discretizer(y) for y in preds]

        return y_predicted

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []

        for i in range(len(X_test)):
            x_test = X_test[i]
            sample_dists = []
        
            for j in range(len(self.X_train)):
                x_train = self.X_train[j]
                total = 0.0
                for k in range(len(x_train)):
                    diff = x_test[k] - x_train[k]
                    total += diff * diff
                    
                distance = total ** 0.5
                sample_dists.append([distance, j])

            for l in range(len(sample_dists)):
                for m in range(l + 1, len(sample_dists)):
                    if sample_dists[l][0] > sample_dists[m][0] or (sample_dists[l][0] == sample_dists[m][0] and sample_dists[l][1] > sample_dists[m][1]):
                        temp = sample_dists[l]
                        sample_dists[l] = sample_dists[m]
                        sample_dists[m] = temp

            k_nearest = []
            count = 0
            while count < self.n_neighbors and count < len(sample_dists):
                k_nearest.append(sample_dists[count])
                count += 1
    
            current_dists = []
            current_indices = []
            for n in range(len(k_nearest)):
                current_dists.append(k_nearest[n][0])
                current_indices.append(k_nearest[n][1])

            distances.append(current_dists)
            neighbor_indices.append(current_indices)

        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        y_predicted = []

        distances, neighbor_indices = self.kneighbors(X_test)

        for i in range(len(neighbor_indices)):
            neighbor_labels = []

            for j in range(len(neighbor_indices[i])):
                index = neighbor_indices[i][j]
                neighbor_labels.append(self.y_train[index])

            label_counts = {}
            for label in neighbor_labels:
                if label not in label_counts:
                    label_counts[label] = 1
                else:
                    label_counts[label] += 1

            most_common_label = None
            highest_count = 0
            for label in label_counts:
                if label_counts[label] > highest_count:
                    highest_count = label_counts[label]
                    most_common_label = label

            y_predicted.append(most_common_label)

        return y_predicted

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        counts = {}
        for label in y_train:
            if label in counts:
                counts[label] += 1
            else:
                counts[label] = 1

        max_count = -1
        most_common = None
        for label in counts:
            if counts[label] > max_count:
                max_count = counts[label]
                most_common = label

        self.most_common_label = most_common

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        
        for _ in X_test:
            y_predicted.append(self.most_common_label)

        return y_predicted


class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(dict): The prior probabilities computed for each
            label in the training set.
        conditionals(dict): The conditional probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.means = None
        self.stdevs = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the conditional probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and conditionals.
        """
        n = len(y_train)

        self.priors = {}
        for y in y_train:
            self.priors[y] = self.priors.get(y, 0) + 1
        for y in self.priors:
            self.priors[y] /= n

        data_by_class = {}
        for x, y in zip(X_train, y_train):
            if y not in data_by_class:
                data_by_class[y] = []
            data_by_class[y].append(x)

        self.means = {}
        self.stdevs = {}

        for y in data_by_class:
            rows = data_by_class[y]
            cols = list(zip(*rows))   

            self.means[y] = [sum(col) / len(col) for col in cols]

            self.stdevs[y] = [
                (sum((val - mean)**2 for val in col) / len(col)) ** 0.5
                for col, mean in zip(cols, self.means[y])
            ]

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_pred = []

        for row in X_test:
            posteriors = {}

            for y in self.priors:
                posterior = myutils.math.log(self.priors[y])

                for i, x in enumerate(row):
                    mean = self.means[y][i]
                    sdev = self.stdevs[y][i]
                    likelihood = myutils.gaussian(x, mean, sdev)
                    posterior += myutils.math.log(likelihood + 1e-9)

                posteriors[y] = posterior

            best_label = max(posteriors, key=posteriors.get)
            y_pred.append(best_label)

        return y_pred
    
class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train
        self.tree = myutils.build_tree(X_train, y_train, list(range(len(X_train[0]))))


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        return [myutils.predict_one(x, self.tree) for x in X_test]

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        def recurse(node, path):
            if node[0] == "Leaf":
                rule = "IF " + " AND ".join(path) if path else "IF True"
                print(f"{rule} THEN {class_name} = {node[1]}")
                return
            attr_index = int(node[1][3:])
            attr_name = attribute_names[attr_index] if attribute_names else node[1]
            for val_branch in node[2:]:
                val = val_branch[1]
                subtree = val_branch[2]
                recurse(subtree, path + [f"{attr_name} == {val}"])
        recurse(self.tree, [])

        pass


class MyRandomForestClassifier:
    def __init__(self, N=5, M=3, F=None, random_state=None):
        self.X_train = None
        self.y_train = None

        self.N = N
        self.M = M
        self.F = F

        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.trees = []
        self.oob_acc = []
        self.att_subsets = []

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        n_samples = len(X_train)
        n_features = len(X_train[0])

        if self.F is None:
            self.F = max(1, int(np.ceil(np.sqrt(n_features))))

        self.trees = []
        all_attributes = list(range(n_features))

        for _ in range(self.N):

            if self.random_state is None:
                seed = None
            else:
                seed = self.random_state + _

            X_boot, X_oob, y_boot, y_oob = myevaluation.bootstrap_sample(X_train, y_train, n_samples=n_samples, random_state=seed)

            att_subset = list(self.rng.choice(all_attributes, size=self.F, replace=False))
            self.att_subsets.append(att_subset)

            tree = MyDecisionTreeClassifier()

            #manual fit
            tree.X_train = X_boot
            tree.y_train = y_boot
            tree.tree = myutils.build_tree(X_boot, y_boot, att_subset)

            self.trees.append(tree)

            #obb accuracy
            if len(X_oob) == 0:
                self.oob_acc.append(0.0)
            else:
                preds = tree.predict(X_oob)
                correct = sum(1 for p, t in zip(preds, y_oob) if p == t)
                acc = correct / len(y_oob)
                self.oob_acc.append(acc)

        #select best trees
        sorted_indices = np.argsort(self.oob_acc)[::-1]
        self.selected_indices = sorted_indices[:self.M].tolist()

    def predict(self, X_test):
        
        predictions = []

        #predictions from each tree
        for index in self.selected_indices:
            tree = self.trees[index]
            preds = tree.predict(X_test)
            predictions.append(preds)

        predictions = list(zip(*predictions))

        #list of final predictions of all test samples
        final = []
        for votes in predictions:
            c = Counter(votes)
            final.append(c.most_common(1)[0][0])

        return final