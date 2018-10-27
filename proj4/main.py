#!/usr/bin/env python3.6
from sklearn.datasets import load_digits

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree         import DecisionTreeRegressor

import numpy as np
import argparse
import timeit
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from pandas import read_csv
from sklearn.metrics import accuracy_score


def getDataset(dataset_name):
    """ helper function for main to load datasets
        given the args input

        dataset_name: string
            'digits' or 'realdisp'
    """
    if dataset_name.lower().endswith(('.data.txt', '.data')):
        # assume this data is whitespace delimited 
        df = read_csv(dataset_name,
                      delim_whitespace=True,
                      header=None,
                      prefix='x')

    else:  # assume it is a normal csv w/ header
        df = read_csv(dataset_name)

    # assume last col is target
    X = df.iloc[:, [0,-2]]
    y = df.iloc[:, -1]

    return X, y


def getClassifier(classifier_name):
    """ helper function for main to get the right classifier
        given the args input

        classifier_name: string
            'linreg', 'ransac', 'ridge', 'lasso'
    """
    if classifier_name == 'linreg':
        cls = LinearRegression()

    elif classifier_name == 'ransac':
        cls = RANSACRegressor()

    elif classifier_name == 'ridge':
        cls = Ridge(alpha=1.0)

    elif classifier_name == 'lasso':
        cls = Lasso(alpha=1.0)

    elif classifier_name == 'nonlinear':
        cls = DecisionTreeRegressor(max_depth=3)

    return cls


def normalEquation(X, y):
    """ Returns slope, intercept
    """
    onevec = np.ones((X.shape[0]))
    onevec = onevec[:, np.newaxis]
    Xb = np.hstack((onevec, X))
    
    w = np.zeros(X.shape[1])
    z = np.linalg.inv(np.dot(Xb.T, Xb))
    w = np.dot(z,np.dot(Xb.T, y))

    # for plotting
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = w[0] + w[1] * x_vals
    plt.plot(x_vals, y_vals, '--')

    return w[1], w[0]


def simplePlot(X, y_train_pred, y_test_pred, y_test, y_train):
    """ Plot the train and test data

        classifier_name: string
            'linreg', 'ransac', 'ridge', 'lasso'
    """
    plt.scatter(y_train_pred, y_train_pred-y_train, c='steelblue', marker='o',
                label='Training data')
    plt.scatter(y_test_pred, y_test_pred-y_test, c='green', marker='s',
                label='Test data')
    plt.legend(loc="lower left")
    plt.xlim([-20,60])
    
    plt.show()


if __name__=='__main__':
    # set/handle arguments
    parser = argparse.ArgumentParser(
                description='Run classifiers on data (80:20 train:test)'
            )
    parser.add_argument('classifier', 
                        help='The classifier to use',
                        choices=['linreg', 'ransac', 'ridge', 'lasso', 
                                 'nonlinear']
                        )
    parser.add_argument('dataset', 
                        help='The dataset filename, or digits/realdisp', 
                        )
    args = parser.parse_args()
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
    y_train_pred = cls.predict(X_train)
    y_test_pred  = cls.predict(X_test)
    stop  = timeit.default_timer()
    print("Test Time (s): " + str(stop-start))

    # MSE 
    error_train = mean_squared_error(y_train, y_train_pred)
    error_test  = mean_squared_error(y_test,  y_test_pred)
    print('MSE train: %.3f, test: %.3f' % (error_train, error_test))

    # R^2 
    r2_train = r2_score(y_train, y_train_pred)
    r2_test  = r2_score(y_test,  y_test_pred)
    print('R^2 train: %.3f, test: %.3f' % (r2_train, r2_test))

    # Plot
    simplePlot(X, y_train_pred, y_test_pred, y_test, y_train)


    # Calc Accuracy
    #acc = accuracy_score(y_test, y_pred)
    #print("Accuracy: " + str(acc))
