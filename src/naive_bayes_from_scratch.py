# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 15:27:07 2019

@author: Binary
"""
'''
This file provides the class for Bernaoulli Naive Bayes from scratch
'''
import numpy as np  
from sklearn.base import BaseEstimator

def BernoulliNB(X_train, y_train, X_test, y_test, X_test_tfidf):
    
    print("Press 1 for laplace smoothing and 0 otherwise")
    s = input("Enter the selection: ")
    s = int(s)
    if s==1:
        LS = True
    else:
        LS = False
        
    class BernoulliNBFromScratch(BaseEstimator):      
        def __init__(self, l_smooth = LS):
            """
            Implementation of Bernoulli Naive Bayes from scratch
            """
            self.l_smooth = l_smooth
            
        def fit(self, X_train, y_train):
            """
            This function fits preprocessed data with the target variable
            """
            
            # Number of examples where y = 0,1
            No_y_train_1 = np.sum(y_train)
            No_y_train_0 = y_train.shape[0] - No_y_train_1
            
            #Ratio of Number of examples where y=0,1 and the total number of examples
            self.theta_0 = No_y_train_0/y_train.shape[0]
            self.theta_1 = No_y_train_1/y_train.shape[0]
                   
            #Ratio of Number of examples where x_j =1 and y=0,1 and Number of examples where y=0,1 respectively
            No_inst_j1 = X_train.T.dot(y_train.reshape([-1,1]))  
            No_inst_j0 = X_train.T.dot(1-y_train.reshape([-1,1]))
            
            #Whether or not laplace smoothing is implemented  or not
            if self.l_smooth:
                self.prob1 = (No_inst_j1 + 1)/(No_y_train_1 + 2)
                self.prob0 = (No_inst_j0 + 1)/(No_y_train_0 + 2)
            else:
                self.prob1 = No_inst_j1/No_y_train_1
                self.prob0 = No_inst_j0/No_y_train_0
            
            return self
    
        def predict(self, X_test):
            '''
            This function gives the predicted values depending on the X values
            ''' 
            #Calculation of Prediction: delta function
            del_x = (np.log(self.theta_1/(1-self.theta_1)) + X_test.toarray().dot(np.log(self.prob1/self.prob0)) + (1-X_test.toarray()).dot(np.log((1-self.prob1)/(1-self.prob0))))
            
            # If delta function is greater than 0, classify as 1, otherwise as 0
            pred_val = np.zeros(del_x.shape).astype(dtype = 'int')
            pred_val[del_x > 0] = 1
            
            return pred_val.reshape([-1,])
    
        def accuracy(self, X_test, y_test, X_test_tfidf):
            '''
            This function gives the accuracy value of the predicted 
            '''
            assert np.array_equal(y_test, y_test.astype(dtype = bool))
            
            # y validation is predicted
            ypred = self.predict(X_test)
            
            #y values for test set is predicted
            y_pred_test = self.predict (X_test_tfidf)
    
            return ypred, y_pred_test

    
    bnb = BernoulliNBFromScratch(l_smooth = LS) 
    bnb.fit(X_train, y_train.values)
    ypred, y_pred_test = bnb.accuracy(X_test, y_test, X_test_tfidf ) 
    return ypred, y_pred_test 
