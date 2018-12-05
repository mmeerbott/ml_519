#/usr/bin/env python3.6

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn import linear_model
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



def main ():
    dataset = pd.read_csv('../../data.csv', header=0)
    X = dataset.iloc[:, 2:18].values
    y = dataset.iloc[:, 18].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    lda = LDA(n_components=2)
    X_train = lda.fit_transform(X_train, y_train)
    X_test = lda.transform(X_test)
    print("---------Perceptron---------")
    per = Perceptron(n_iter=50, eta0=.1, random_state=1)
    per.fit(X_train, y_train)
    y_pred1 = per.predict(X_test)
    accuracy1 = ((y_test==y_pred1).sum()/len(y_test)*100)
    print('accuracy %.2f' %accuracy1)
    print("---------Decision Tree---------")
    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=1, max_depth=5, min_samples_leaf=3)
    clf_entropy.fit(X_train, y_train)
    y_pred2 = clf_entropy.predict(X_test)
    accuracy2 = ((y_test==y_pred2).sum()/len(y_test)*100)
    print('accuracy %.2f' % accuracy2)
    print("---------KNN---------")
    knnn = KNeighborsClassifier(n_neighbors=9, metric='euclidean')  # how did i choose
    #print(len(X_test))
    knnn.fit(X_train, y_train)
    y_pred3 = knnn.predict(X_test)
    accuracy3 = ((y_test==y_pred3).sum()/len(y_test)*100)
    print('accuracy %.2f' % accuracy3)
    print("---------Logestic Refression---------")
    logreg = LogisticRegression(multi_class='auto')
    logreg.fit(X_train, y_train)
    y_pred4 = logreg.predict(X_test)
    #y_pred_lr_prob = logreg.predict_log_proba(X_test)
    #print(y_pred_lr_prob.shape)
    #print(y_pred_lr_prob)
    accuracy4 = ((y_test==y_pred4).sum()/len(y_test)*100)
    print('accuracy %.2f' % accuracy4)
    print("---------SVM Linear---------")
    clf1 = svm.SVC(kernel="linear", random_state=1, C=1)
    clf1.fit(X_train, y_train)
    y_pred5 = clf1.predict(X_test)
    accuracy5 = ((y_test==y_pred5).sum()/len(y_test)*100)
    print('accuracy %.2f' % accuracy5)
    print("---------SVM non-Linear---------")
    #clf = svm.SVC(kernel='rbf', random_state=1, gamma='auto', C=20.0)
    clf2 = svm.SVC(gamma='scale', C=1.0)
    clf2.fit(X_train, y_train)
    y_pred6 = clf2.predict(X_test)
    accuracy6 = ((y_test==y_pred6).sum()/len(y_test)*100)
    print('accuracy %.2f' % accuracy6)

    print("---------SGD---------")
    sgd = linear_model.SGDClassifier(max_iter=100, tol=1e-3)
    sgd.fit(X_train, y_train)
    y_pred7 = sgd.predict(X_test)
    accuracy7 = ((y_test == y_pred7).sum() / len(y_test) * 100)
    print('accuracy %.2f' % accuracy7)
main()
