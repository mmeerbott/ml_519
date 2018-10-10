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


def getDataset(dataset_name):
    """ helper function for main to load one of two datasets
        given the args input

        dataset_name: string
            'digits' or 'realdisp'
    """
    if dataset_name == 'digits':
        X, y = load_digits(return_X_y=True)

    elif dataset_name == 'realdisp':
        df = read_csv('datasets/realdisp/subject1_ideal.log',
                       sep='\t',
                       header=None,
                       prefix='x')
        X = df.iloc[:, [0,118]]
        y = df.iloc[:, 119]

    else:  # assume it is a normal csv w/ header, last col is target
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


if __name__=='__main__':
    # set/handle arguments
    parser = argparse.ArgumentParser(
                 description='Run classifiers on data(.80 to train .20 to test)'
             )

    parser.add_argument('classifier', 
                        help='The classifier to use',
                        choices=['perceptron', 'linsvm', 'nonlinsvm',
                                 'dtree', 'knn', 'logreg']
                        )

    parser.add_argument('dataset', 
                        help='The dataset filename, or digits/realdisp', 
                        )

    args = parser.parse_args()

    # Get classifier and dataset
    cls = getClassifier(args.classifier)
    X, y = getDataset(args.dataset)

    # Split the train:test data 80:20
    training_size = int(len(X)*.8)
    print(args.classifier + ' ' + args.dataset)
    print("Training Size (instances): " + str(training_size))
    print("Testing Size (instances): " + str(len(X) - training_size))

    X_train = X[0:training_size]
    y_train = y[0:training_size]

    X_test = X[training_size:]
    y_test = y[training_size:]

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
