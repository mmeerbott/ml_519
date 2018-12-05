#!/usr/bin/env python3.6

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
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
import os




dataset = pd.read_csv('data.csv', header=0)
X = dataset.iloc[:, 2:18].values
y = dataset.iloc[:, 18].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3, stratify=y)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
lda_number = [4,5,6,7,8,9,10,11,12]
for x in lda_number:
	lda = LDA(n_components=x)
	X_train = lda.fit_transform(X_train, y_train)
	X_test = lda.transform(X_test)
	print("--------- LDA = ", x , " ---------")
	print("---------Perceptron---------")
#per = Perceptron(n_iter=50, eta0=.1, random_state=1)
#per.fit(X_train, y_train)
#y_pred1 = per.predict(X_test)
#accuracy1 = ((y_test==y_pred1).sum()/len(y_test)*100)
#print('accuracy %.2f' %accuracy1)
	pipe_perceptron = make_pipeline(StandardScaler(),Perceptron(random_state=1))
	param_range_1=[5,50,100,200,400,1000,2000] 
	param_range_2=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
	param_grid = [{'perceptron__max_iter': param_range_1, 'perceptron__eta0': param_range_2 }]
	gs = GridSearchCV(estimator=pipe_perceptron,param_grid=param_grid,scoring='accuracy',cv=3,n_jobs=-1)
	gs = gs.fit(X_train, y_train)

	clf = gs.best_estimator_
	clf.fit(X_train, y_train)
	print('Train accuracy: %.3f' % gs.best_score_)
	print('Best Parameter: ', gs.best_params_)
	print('Test accuracy: %.3f' % clf.score(X_test, y_test))


	print("---------Decision Tree---------")
#clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=1, max_depth=5, min_samples_leaf=3)
#clf_entropy.fit(X_train, y_train)
#y_pred2 = clf_entropy.predict(X_test)
#accuracy2 = ((y_test==y_pred2).sum()/len(y_test)*100)
#print('accuracy %.2f' % accuracy2)


	decision_tree_classifier = DecisionTreeClassifier()

	parameter_grid = {'max_depth': [1, 2, 3, 4, 5,6,7,8,9,10], 'min_samples_leaf': [1, 2, 3, 4,5,6,7,8,9,10]}



	gs = GridSearchCV(decision_tree_classifier, param_grid = parameter_grid,cv = 3)
	gs.fit(X_train, y_train)

	clf = gs.best_estimator_
	clf.fit(X_train, y_train)
	print('Train accuracy: %.3f' % gs.best_score_)
	print('Best Parameter: ', gs.best_params_)
	print('Test accuracy: %.3f' % clf.score(X_test, y_test))


#pipe_dt = make_pipeline(StandardScaler(),DecisionTreeClassifier(random_state=1))
#param_range_1=[5] 
#param_range_2=[3]
#param_grid = [{'dt__max_depth': param_range_1, 'dt__min_samples_leaf': param_range_2 }]
#gs = GridSearchCV(DecisionTreeClassifier(),param_grid=param_grid,scoring='accuracy',cv=3,n_jobs=-1)
#gs = gs.fit(X_train, y_train)

#X_train , X_test , y_train , y_test = train_test_split (X, y, test_size=0.3, random_state=1, stratify=y)
#clf = gs.best_estimator_
#clf.fit(X_train, y_train)
#print('Train accuracy: %.3f' % gs.best_score_)
#print('Best Parameter: ', gs.best_params_)
#print('Test accuracy: %.3f' % clf.score(X_test, y_test))

	print("---------KNN---------")
#knnn = KNeighborsClassifier(n_neighbors=9, metric='euclidean') 
	knn = KNeighborsClassifier() # how did i choose

	parameter_grid = {'n_neighbors': [1, 2, 3, 4, 5,6,7,8,9,10], 'metric': ['minkowski', 'euclidean']}



	gs = GridSearchCV(knn, param_grid = parameter_grid,cv = 3)
	gs.fit(X_train, y_train)

	clf = gs.best_estimator_
	clf.fit(X_train, y_train)
	print('Train accuracy: %.3f' % gs.best_score_)
	print('Best Parameter: ', gs.best_params_)
	print('Test accuracy: %.3f' % clf.score(X_test, y_test))
#print(len(X_test))
#knnn.fit(X_train, y_train)
#y_pred3 = knnn.predict(X_test)
#accuracy3 = ((y_test==y_pred3).sum()/len(y_test)*100)
#print('accuracy %.2f' % accuracy3)
#print("---------Logestic Refression---------")
#logreg = LogisticRegression(multi_class='auto')
#logreg.fit(X_train, y_train)
#y_pred4 = logreg.predict(X_test)
#y_pred_lr_prob = logreg.predict_log_proba(X_test)
#print(y_pred_lr_prob.shape)
#print(y_pred_lr_prob)
#accuracy4 = ((y_test==y_pred4).sum()/len(y_test)*100)

	print("---------Logestic Regression---------")
	logreg = LogisticRegression(multi_class='auto')
	parameter_grid = { 'tol': [0.0001, 0.001, 0.01, 0.1,1.0],'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 'max_iter': [50,100,200,500,1000,200,5000,10000]}
	gs = GridSearchCV(logreg, param_grid = parameter_grid,cv = 3)
	gs.fit(X_train, y_train)

	clf = gs.best_estimator_
	clf.fit(X_train, y_train)
	print('Train accuracy: %.3f' % gs.best_score_)
	print('Best Parameter: ', gs.best_params_)
	print('Test accuracy: %.3f' % clf.score(X_test, y_test))
	print("--------- SVM ---------")
	svc = svm.SVC(kernel="linear", random_state=1, C=1)
	parameter_grid = [{'C': [0.0001, 0.001, 0.01, 0.1,1.0, 10.0, 50,100,200,500,1000], 'gamma': [0.0001, 0.001, 0.01, 0.1,1.0, 10.0, 50,100,200,500,1000],'kernel': ['rbf']}, {'C': [0.0001, 0.001, 0.01, 0.1,1.0, 10.0, 50,100,200,500,1000],'kernel': ['linear']}]
	gs = GridSearchCV(svc, param_grid = parameter_grid,cv = 3)
	gs.fit(X_train, y_train)

	clf = gs.best_estimator_
	clf.fit(X_train, y_train)
	print('Train accuracy: %.3f' % gs.best_score_)
	print('Best Parameter: ', gs.best_params_)
	print('Test accuracy: %.3f' % clf.score(X_test, y_test))

	print("---------SGD---------")
	sgd = linear_model.SGDClassifier()
	parameter_grid = {'penalty': ['none', 'l2', 'l1','elasticnet'], 'max_iter': [5, 10,50,100, 250, 500, 750, 1000, 1500, 3000, 6000, 10000], 'tol': [0.0001, 0.001, 0.01 ,0.05, 0.1] }
#parameter_grid = {'penalty': ['none', 'l2', 'l1','elasticnet'], 'max_iter': [5, 10,50,100, 250, 500, 750, 1000, 1500, 3000, 6000, 10000],'tol': [0.0001, 0.001, 0.01 0.05 0.1], 'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']}
	gs = GridSearchCV(sgd, param_grid = parameter_grid,cv = 3)
	gs.fit(X_train, y_train)

	clf = gs.best_estimator_
	clf.fit(X_train, y_train)
	print('Train accuracy: %.3f' % gs.best_score_)
	print('Best Parameter: ', gs.best_params_)
	print('Test accuracy: %.3f' % clf.score(X_test, y_test))
    

