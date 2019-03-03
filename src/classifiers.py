# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 13:05:37 2019

@author: Binary
"""
'''
This file has the list of the functions of the different classifiers.
'''
#Importing libraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
import warnings
warnings.filterwarnings("ignore")

#Classifiers

#Logisitic Regression
def logisticreg(X_train, y_train, X_test, y_test, X_test_tfidf):
    param_grid = {'C': [1, 10, 100, 1000 ]}
    grid = GridSearchCV(LogisticRegression(solver='lbfgs',multi_class='multinomial',random_state=0), param_grid, cv=10)
    grid.fit(X_train, y_train)
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))
    print("Best parameters: ", grid.best_params_)
    print("Best estimator: ", grid.best_estimator_)
    lr          = grid.best_estimator_
    ypred       = lr.predict(X_test)
    y_pred_test = lr.predict(X_test_tfidf)
    return ypred, y_pred_test

#RidgeClassifier
def ridgeclassifier(X_train, y_train, X_test, y_test, X_test_tfidf): 
    param_grid = {'alpha': [0.01, 0.1, 0.5, 1, 2]}
    grid = GridSearchCV(RidgeClassifier(class_weight = 'balanced'), param_grid, cv=10)
    grid.fit(X_train, y_train)
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))
    print("Best parameters: ", grid.best_params_)
    print("Best estimator: ", grid.best_estimator_)
    rc          = grid.best_estimator_
    ypred       = rc.predict(X_test)
    y_pred_test = rc.predict(X_test_tfidf)
    return ypred, y_pred_test

#Decision tree
def decisiontree(X_train, y_train, X_test, y_test, X_test_tfidf):
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100)
    clf_entropy.fit(X_train, y_train)
    ypred       = clf_entropy.predict(X_test)
    y_pred_test = clf_entropy.predict(X_test_tfidf)
    return ypred, y_pred_test

#Support Vector Machine
def linearSVC(X_train, y_train, X_test, y_test, X_test_tfidf):
    Cs = [0.01, 0.1, 0.5, 1, 1.5,2, 2.5, 3 ]
    svc_clf = GridSearchCV(LinearSVC(), param_grid=dict(C=Cs), cv=10)
    svc_clf.fit(X_train, y_train)    
    print("Best cross-validation score: {:.2f}".format(svc_clf.best_score_))
    print("Best parameters: ", svc_clf.best_params_)
    print("Best estimator: ", svc_clf.best_estimator_)
    svc_clf     = svc_clf.best_estimator_
    ypred       = svc_clf.predict(X_test)
    y_pred_test = svc_clf.predict(X_test_tfidf)
    return ypred, y_pred_test

#Stochastic Gradient Descent
def stochastic_GD(X_train, y_train, X_test, y_test, X_test_tfidf):
    param_grid = {'alpha': [1e-4, 1e-5, 1e-6]}
    grid = GridSearchCV(SGDClassifier(epsilon=0.1), param_grid, cv=10)
    grid.fit(X_train, y_train)
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))
    print("Best parameters: ", grid.best_params_)
    print("Best estimator: ", grid.best_estimator_)
    sgd         = grid.best_estimator_
    ypred       = sgd.predict(X_test)
    y_pred_test = sgd.predict(X_test_tfidf)
    return ypred, y_pred_test

#Multinomial Naive Bayes model
def MNB(X_train, y_train, X_test, y_test, X_test_tfidf):
    param_grid = {'alpha': [0.01, 0.1, 1, 1.5]}
    grid = GridSearchCV(MultinomialNB(), param_grid, cv=10)
    grid.fit(X_train, y_train)
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))
    print("Best parameters: ", grid.best_params_)
    print("Best estimator: ", grid.best_estimator_)
    mnb        = grid.best_estimator_
    ypred      = mnb.predict(X_test)
    y_pred_test= mnb.predict(X_test_tfidf)
    return ypred, y_pred_test

#Naive Bayes Support vector machine
#Beware of the memory
def nbsvm(X_train, y_train, X_test, y_test, X_test_tfidf):
    import numpy as np
    from scipy.sparse import spmatrix, coo_matrix
    from sklearn.base import BaseEstimator
    from sklearn.linear_model.base import LinearClassifierMixin, SparseCoefMixin
    from sklearn.svm import LinearSVC
    
    class NBSVM(BaseEstimator, LinearClassifierMixin, SparseCoefMixin):
    
        def __init__(self, alpha=2, C=3, beta=1, fit_intercept=False):
            self.alpha = alpha
            self.C = C
            self.beta = beta
            self.fit_intercept = fit_intercept
    
        def fit(self, X, y):
            self.classes_ = np.unique(y)
            if len(self.classes_) == 2:
                coef_, intercept_ = self._fit_binary(X, y)
                self.coef_ = coef_
                self.intercept_ = intercept_
            else:
                coef_, intercept_ = zip(*[
                    self._fit_binary(X, y == class_)
                    for class_ in self.classes_
                ])
                self.coef_ = np.concatenate(coef_)
                self.intercept_ = np.array(intercept_).flatten()
            return self
    
        def _fit_binary(self, X, y):
            p = np.asarray(self.alpha + X[y == 1].sum(axis=0)).flatten()
            q = np.asarray(self.alpha + X[y == 0].sum(axis=0)).flatten()
            r = np.log(p/np.abs(p).sum()) - np.log(q/np.abs(q).sum())
            b = np.log((y == 1).sum()) - np.log((y == 0).sum())
    
            if isinstance(X, spmatrix):
                indices = np.arange(len(r))
                r_sparse = coo_matrix(
                    (r, (indices, indices)),
                    shape=(len(r), len(r))
                )
                X_scaled = X * r_sparse
            else:
                X_scaled = X * r
    
            lsvc = LinearSVC(
                C=self.C,
                fit_intercept=self.fit_intercept,
                max_iter=10000
            ).fit(X_scaled, y)
    
            mean_mag =  np.abs(lsvc.coef_).mean()
    
            coef_ = (1 - self.beta) * mean_mag * r + \
                    self.beta * (r * lsvc.coef_)
    
            intercept_ = (1 - self.beta) * mean_mag * b + \
                         self.beta * lsvc.intercept_
    
            return coef_, intercept_
    mnbsvm = NBSVM()
    mnbsvm.fit(X_train.toarray(), y_train)
    ypred = mnbsvm.predict(X_test.toarray())
    y_pred_test=mnbsvm.predict(X_test_tfidf)
    return ypred, y_pred_test