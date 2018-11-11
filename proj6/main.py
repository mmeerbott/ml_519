#!/usr/bin/env python3.6

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LR

from sklearn import datasets

import numpy as np
import argparse
import timeit
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def setupArgs():
    """ set/handle arguments
    """
    parser = argparse.ArgumentParser(
                description='Run classifiers on data (80:20 train:test)'
            )
    parser.add_argument('classifier', 
                        help='The classifier to use',
                        choices=['pca', 'lda', 'kpca']
                        )
    parser.add_argument('dataset', 
                        help='The dataset filename, or mnist', 
                        )
    args = parser.parse_args()
    return args


def fitPredictClustering(cls_name, X):
    """ helper function for main to get the right clustering algo
        given the args input. Runs the Algo and returns labels

        classifier_name: string
            'kmeans', 'hsklearn', 'hscipy', 'dbscan'
    """
    if cls_name == 'kmeans':
        cls = KMeans(n_clusters=3, max_iter=1000).fit_predict(X)

    elif cls_name == 'hsklearn':
        cls = AgglomerativeClustering(n_clusters=3).fit_predict(X)

    elif cls_name == 'hscipy':
        Z = linkage(X, method='complete', metric='euclidean')
        cls = fcluster(Z, 2.5, criterion='distance')

    elif cls_name == 'dbscan':
        cls = DBSCAN(eps=0.7, min_samples=5, metric='euclidean').fit_predict(X)

    return cls


def dimReduce(method, X, y):
    if method == 'lda':
        res = LDA(n_components=2)
        res.fit_transform(X, y)
    
    if method == 'pca':
        pca = PCA(n_components=None)
        # std'ize X
        sc = StandardScaler()
        X_std = sc.fit_transform(X)
        res = pca.fit_transform(X_std)
   
    if method == 'kpca':
        kpca = KernelPCA(n_components=2, kernel='rbf')
        res = kpca.fit_transform(X)

    return res


def accuracyLogReg(X_train, y_train, X_test):
    lr = LR(solver='lbfgs', multi_class='multinomial')
    lr.fit(X_train, y_train)
    y_test_pred = lr.predict(X_test)
    accuracy = sum(y_test_pred == y_test) / len(y_test)
    return accuracy


def getDataset(dataset_name):
    """ helper function for main to load datasets
        given the args input

        dataset_name: string
            'digits' or 'realdisp'
    """
    if dataset_name == 'mnist':
        data = datasets.load_digits()
        return data.data, data.target

    elif dataset_name.lower().endswith(('.data.txt', '.data')):
        # assume this data is whitespace delimited 
        df = pd.read_csv(dataset_name,
                      delim_whitespace=True,
                      header=None,
                      prefix='x')

    else:  # assume it is a normal csv w/ header
        df = pd.read_csv(dataset_name)

    # fill missing data
    df.fillna(df.mean(), inplace=True)

    # assume last col is target
    X = df.iloc[:, [0,-2]]
    y = df.iloc[:, -1]

    return X, y


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


def train(cls, X_train, y_train):
    """ train and time it 
    """
    start = timeit.default_timer()
    cls.fit(X_train, y_train)
    stop  = timeit.default_timer()
    print("Train Time (s): " + str(stop-start))
    return cls


def test(cls, X_train, X_test):
    """ test and time it 
    """
    start = timeit.default_timer()
    y_train_pred = cls.predict(X_train)
    y_test_pred  = cls.predict(X_test)
    stop  = timeit.default_timer()
    print("Test Time (s): " + str(stop-start))


def kmeans_elbow(X):
    """ plot kmeans to check for a visual 'elbow' to choose k 
    """
    sse = {}
    kmeans = [KMeans(n_clusters=k, max_iter=1000).fit(X).score(X) for k in range(1,10)]
        # sse[k] = kmeans.inertia_
    plt.figure()
    plt.plot(range(len(kmeans)), kmeans)
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.show()


def getMinptsE(X):
    K = len(X)-1
    n = len(X)
    nn = NearestNeighbors(n_neighbors=(K+1))
    nbrs = nn.fit(X)
    distances, indices = nbrs.kneighbors(X)
    print(indices)
    print(distances)

    distK = np.empty([K,n])
    for i in range(K):
        distance_Ki = distances[:, (i+1)]
        distance_Ki.sort()
        distnace_Ki = distance_Ki[::-1]
        distK[i] = distance_Ki
    print(distance_Ki)

    for i in range(K):
        plt.plot(distK[i], label='K=%d' %(i+1))
    plt.ylabel('dist')
    plt.xlabel('points')
    plt.legend()
    plt.show()


def getNumericalErrors(y, y_pred):
    """ get MSE and R^2 """
    error = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print('MSE: %.3f' % (error))
    print('R^2: %.3f' % (r2))
    return error, r2


def splitTrainTest(X):
    # Split the train:test data 80:20
    training_size = int(len(X)*.8)
    print(args.classifier + ' ' + args.dataset)
    print("Training Size (instances): " + str(training_size))
    print("Testing Size (instances): " + str(len(X) - training_size))
    return training_size


if __name__=='__main__':
    args = setupArgs()
    # Get classifier and dataset
    X, y = getDataset(args.dataset)

    split = splitTrainTest(X)
    X_train = X[0:split]
    y_train = y[0:split]
    X_test  = X[split:]
    y_test  = y[split:]

    start = timeit.default_timer()
    res   = dimReduce(args.classifier, X_test, y_test)
    stop  = timeit.default_timer()

    acc = accuracyLogReg(X_train, y_train, X_test)
    print("Run Time (s): " + str(stop-start))
    print("Accuracy: " + str(acc))
    print('\n')
