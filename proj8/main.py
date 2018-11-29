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

from sklearn.ensemble  import RandomForestClassifier
from sklearn.ensemble  import BaggingClassifier
from sklearn.ensemble  import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder


def setupArgs():
    """ set/handle arguments
    """
    parser = argparse.ArgumentParser(
                description='Run classifiers on data (80:20 train:test)'
            )
    parser.add_argument('classifier', 
                        help='The classifier to use',
                        choices=['bagging', 'randforest', 'adaboost']
                        )
    parser.add_argument('dataset', 
                        help='The dataset filename, or digits', 
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
    if dataset_name == 'digits':
        data = datasets.load_digits()
        return data.data, data.target

    else:  # assume it is a normal csv w/ header
        df = pd.read_csv(dataset_name)

    # fill missing data
    df.fillna(df.mean(), inplace=True)

    # assume last col is target
    X = df.iloc[:, [0,-2]]
    y = df.iloc[:, -1]

    # label encoder
    le = LabelEncoder()
    y =  le.fit_transform(y)

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


def do_ensemble(method, X, y):
    # prep train test and do whichever method chosen 
    split = splitTrainTest(X)
    X_train = X[0:split]
    y_train = y[0:split]
    X_test  = X[split:]
    y_test  = y[split:]

    if method == 'bagging':
        do_bagging(X_train, y_train, X_test, y_test)
    elif method == 'randforest':
        do_randforest(X_train, y_train, X_test, y_test)
    elif method == 'adaboost':
        do_adaboost(X_train, y_train, X_test, y_test)


def do_bagging(X_train, y_train, X_test, y_test):
    tree = DecisionTreeClassifier(criterion='entropy', random_state=1, 
                                  max_depth=None)
    bag = BaggingClassifier(base_estimator=tree,
                            n_estimators=500, 
                            max_samples=1.0, 
                            max_features=1.0, 
                            bootstrap=True, 
                            bootstrap_features=False, 
                            n_jobs=1, 
                            random_state=1)

    tree = tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)

    tree_train = accuracy_score(y_train, y_train_pred)
    tree_test = accuracy_score(y_test, y_test_pred)
    print('Decision tree train/test accuracies %.3f/%.3f'
          % (tree_train, tree_test))

    bag = bag.fit(X_train, y_train)
    y_train_pred = bag.predict(X_train)
    y_test_pred = bag.predict(X_test)

    bag_train = accuracy_score(y_train, y_train_pred)
    bag_test = accuracy_score(y_test, y_test_pred)
    print('Bagging train/test accuracies %.3f/%.3f'
          % (bag_train, bag_test))


def do_randforest(X_train, y_train, X_test, y_test):
    forest = RandomForestClassifier(criterion='gini', n_estimators=25, 
                                    random_state=1)

    forest.fit(X_train, y_train)
    y_pred_forest = forest.predict(X_test)
    accuracy = (y_test == y_pred_forest).sum() / len(y_test)
    print('Random Forest Accuracy %.3f' % accuracy)


def do_adaboost(X_train, y_train, X_test, y_test):
    tree = DecisionTreeClassifier(criterion='entropy', random_state=1, 
                                  max_depth=None)

    ada = AdaBoostClassifier(base_estimator=tree,
                             n_estimators=500, 
                             learning_rate=0.1,
                             random_state=1)

    tree = tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)

    tree_train = accuracy_score(y_train, y_train_pred)
    tree_test = accuracy_score(y_test, y_test_pred)
    print('Decision tree train/test accuracies %.3f/%.3f'
          % (tree_train, tree_test))

    ada = ada.fit(X_train, y_train)
    y_train_pred = ada.predict(X_train)
    y_test_pred = ada.predict(X_test)

    ada_train = accuracy_score(y_train, y_train_pred) 
    ada_test = accuracy_score(y_test, y_test_pred) 
    print('AdaBoost train/test accuracies %.3f/%.3f'
          % (ada_train, ada_test))


if __name__=='__main__':
    args = setupArgs()
    # Get classifier and dataset
    X, y = getDataset(args.dataset)

    start = timeit.default_timer()
    res   = do_ensemble(args.classifier, X, y)
    stop  = timeit.default_timer()

    print("Run Time (s): " + str(stop-start))
    print('\n')
