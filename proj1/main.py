#!/usr/bin/env python3.6

""" File for run specific configuration """

import argparse
import os
import pandas
import numpy as np

from perceptron import Perceptron
from adaline import Adaline
from sgd import SGD
from ovr import OVR


def good_bad_wine(datasetcsv, makebinary=False):
    df = pandas.read_csv(datasetcsv)

    # -- For tRaining 
    yr = df.iloc[0:1000, 11].values
    Xr = df.iloc[0:1000, [0,10]].values  # get all data (all columns minus class)
    
    # -- For teSting 
    ys = df.iloc[1000:1500, 11].values
    Xs = df.iloc[1000:1500, [0,10]].values

    if makebinary:
        # quality 5 and under will be 'bad' wine; above, 'good'
        classes = [1, -1]
        yr = np.where(yr <= 5, -1, 1)
        ys = np.where(ys <= 5, -1, 1)
    else:
        classes = list(set(yr))

    return {'train':(Xr, yr), 'test':(Xs, ys), 'classes':classes}


def setosa_versicolor(datasetcsv, makebinary=False):
    # get only setosa and versicolor
    df = pandas.read_csv(datasetcsv)

    # -- For tRaining 
    yr = df.iloc[0:100, 4].values
    Xr = df.iloc[0:100, [0, 2]].values  # get sepal length and petal length


    # -- For teSting 
    ys = df.iloc[100:150, 4].values
    Xs = df.iloc[100:150, [0,2]].values

    if makebinary:
        classes = [1, -1]
        yr = np.where(yr == 'setosa', -1, 1)
        ys = np.where(ys == 'setosa', -1, 1)

    else:
        classes = list(set(yr))

    return {'train':(Xr, yr), 'test':(Xs, ys), 'classes':classes}


def preprocess(dataset, makebinary=False):
    """ tries to preprocess the dataset if it is known """
    if dataset == 'datasets/iris_shuffled.csv':
        return setosa_versicolor(dataset, makebinary)
    elif dataset == 'datasets/winequality-red.csv':
        return good_bad_wine(dataset, makebinary)

    # else, assume the data will instead be preprocessed by the user
    return None


if __name__=='__main__':
    # set/handle arguments
    parser = argparse.ArgumentParser(
                 description='Runs Perceptron/Adaline/SGD/OVR classifier on a dataset'
             )
    parser.add_argument('classifier', 
                        help='[Perceptron/Adaline/SGD/OVR]',
                        choices=['perceptron', 'adaline', 'sgd', 'ovr']
                        )

    parser.add_argument('dataset', help='[/path/to/dataset.zip]')
    parser.add_argument('makebinary', help='Makes Dataset binary', action='store_true')
    args = parser.parse_args()

    # Handle the preprocessesing of known datasets
    if (args.classifier != 'ovr'):
        data = preprocess(args.dataset, makebinary=True)
    else:
        data = preprocess(args.dataset)

    if data is None:
        print('Dataset unrecognized.')
        exit()

    X, y = data['train'] 
    Xs, ys = data['test']

    # Based on input, call classifiers
    if   args.classifier == 'perceptron':
        model = Perceptron(eta=0.1, iters=10)
    elif args.classifier == 'adaline':
        model = Adaline(eta=0.1, iters=10)
    elif args.classifier == 'sgd':
        model = SGD(eta=0.1, iters=10)
    elif args.classifier == 'ovr':
        model = OVR(data['classes'], eta=0.1, iters=10)

    model.fit(X, y)
    res = model.predict(Xs)

    matches = 0
    print(res)
    print(ys)
    for i in range(len(ys)):
        if res[i] == ys[i]:
            matches += 1
    accuracy = float(matches) / float(len(ys))

    print(accuracy)
