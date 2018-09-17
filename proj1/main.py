#!/usr/bin/env python3.6

""" File for run specific configuration """

import argparse
import os
import pandas

def setosa_versicolor():
    # TODO 
    pass


if __name__=='__main__':
    # set/handle arguments
    parser = argparse.ArgumentParser(
                 description='Runs Perceptron/Adaline/SGD/SGD-ovr classifier on a dataset'
             )
    parser.add_argument('classifier', 
                        help='[Perceptron/Adaline/SGD/OVR]',
                        choices=['perceptron', 'adaline', 'sgd', 'ovr']
                        )
    parser.add_argument('dataset', help='[/path/to/dataset.zip]')

    args = parser.parse_args()

    # Check Input -- correct classifier, file extension
    _, extension = os.path.splitext(args.dataset)
    assert extension == '.csv', 'Need a csv file'

    # select only setosa and versicolor
    df = pandas.read_csv(args.dataset)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    # TODO Actually do cases for each arg
    # get sepal length and petal length
    X = df.iloc[0:100, [0, 2]].values

    # TODO Actually do cases for each arg
    ppn = Perceptron(eta=0.1, iters=10)
    ppn.fit(X, y)

