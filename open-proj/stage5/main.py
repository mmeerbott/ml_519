#!/usr/bin/env python3.6
from sklearn.datasets import load_digits

from sklearn.linear_model import Perceptron
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier as DTree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import timeit
import argparse
from pandas import read_csv
from sklearn.metrics import accuracy_score

from adaline import Adaline
from sgd import SGD

def getDataset(dataset_name):
    """ helper function for main to load a dataset given the args input.
        Assumes the last column in the data is the target

        dataset_name: string
            The name of the csv file
    """
    df = read_csv(dataset_name)
    X = df.iloc[:, [0,-2]]
    y = df.iloc[:, -1]

    return X, y


def getClassifier(classifier_name):
    """ helper function for main to get the right classifier
        given the args input

        classifier_name: string
            'perceptron'/'linsvm'/'nonlinsvm'/'dtree'/'knn'/'logreg'
    """
    if classifier_name == 'perceptron':
        cls = Perceptron(tol=1e-3, random_state=0)

    elif classifier_name == 'linsvm':
        cls = svm.SVC(kernel='linear', gamma='scale', C=10)

    elif classifier_name == 'nonlinsvm':
        cls = svm.SVC(kernel='rbf', gamma='scale', C=10)

    elif classifier_name == 'dtree':
        cls = DTree(criterion='gini', max_depth=4)

    elif classifier_name == 'knn':
        cls = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')

    elif classifier_name == 'logreg':
        cls = LogisticRegression(C=100.0, random_state=1)

    return cls


def setupArgs():
    parser = argparse.ArgumentParser(
                 description='Run classifiers on data(.80 to train .20 to test)'
             )

    parser.add_argument('classifier', 
                        help='The classifier to use',
                        choices=['perceptron', 'adaline', 'sgd']
                        )

    parser.add_argument('dataset', 
                        help='The dataset filename, or digits/realdisp', 
                        )

    return parser.parse_args()


def splitTrainTest(X, y, perc_train=.7):
    """ Split where perc_train is the percentage of training data to use """
    training_size = int(len(X) * perc_train)
    X_train = X[0:training_size]
    y_train = y[0:training_size]
    X_test = X[training_size:]
    y_test = y[training_size:]

    return X_train, y_train, X_test, y_test


if __name__=='__main__':
    # Get classifier and dataset
    cls = getClassifier(args.classifier)
    X, y = getDataset(args.dataset)

    # Split the train:test data 
    X_train, y_train, X_test, y_test = splitTrainTest(X, y, .8)

    # Train
    start = timeit.default_timer()
    cls.fit(X_train, y_train)
    stop  = timeit.default_timer()

    print("Train Time (s): " + str(stop-start))
    
    # Test 
    start = timeit.default_timer()
    y_pred = cls.predict(X_test)
    stop  = timeit.default_timer()

    print("Test Time (s): " + str(stop-start))

    # Calc Accuracy
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy: " + str(acc))
