# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 13:12:16 2019

@author: Binary
"""
'''
This file provides the pipeline function that uses two feature extraction pipelines 
for processing the text data (binary occurrences vs. tf-idf weighting)with LinearSVC model.
It also provides us the Model Validation Pipeline.
'''
#Importing the libraries
 
#from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.model_selection import train_test_split  
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

def pipeline_data(cleandata_Train, y_datatrain):
    #Spliting the cleandata into training and validation sets
    x_train, x_validation, y_train, y_validation = train_test_split(cleandata_Train, y_datatrain, test_size=.16, random_state=2000)
    
    #Pipeline for Binary occurances
    binary_occurance = Pipeline([('vect', CountVectorizer() ),
                                  ('clf', LinearSVC() )])
    
    # Pipeline for Tfidf weighting
    tfidf_weighting = Pipeline([('vect', CountVectorizer() ),
                                ('tfidf', TfidfTransformer() ),
                                ('clf', LinearSVC() )]) 
    #Creating a list of pipeline to see which is better    
    Pipeline_list = {
                        "binary_linearSVC": binary_occurance,
                        "tfidf_linearSVC": tfidf_weighting
                      }
    
    # Assigning the parameters of binary Countvectorizer and Linear SVC
    binary_svc_parameters = { 
                              "vect__binary": [True], 
                              "vect__ngram_range": [(1,3)], 
                              "vect__min_df" : [2],
                              #"vect__max_df":[0.8,0.9],
                              "clf__C": [1, 2.5]
                              }
    # Assigning the parameters of Tfidf and Linear SVC
    tfidf_svc_parameters = {       
                              "vect__ngram_range": [(1,3)], #(1,2),(1,3)
                              "tfidf__use_idf": [True],
                              #"vect__max_df":[0.8,0.9],
                              "vect__min_df" : [2],
                              "clf__C": [ 1, 2.5]
                              }
    # Keeping the parameters in a list
    parameters_list = {
                         "binary_linearSVC": binary_svc_parameters,
                         "tfidf_linearSVC" : tfidf_svc_parameters,
                         }
    
    #Perfroming K-fold cross Validaton by Grid Search CV method keeping the value of CV= 10
    for i in Pipeline_list:
        print('Performing Grid Search for {}'.format(i))
        grid = GridSearchCV(Pipeline_list[i], param_grid = parameters_list[i], cv=10)
        grid.fit(x_train, y_train)
        print("Best cross-validation score: {:.2f}".format(grid.best_score_))
        print("Best parameters: ", grid.best_params_)
        print("Best estimator: ", grid.best_estimator_)
        svc = grid.best_estimator_
        acc_svc=accuracy_score(y_validation, svc.predict(x_validation)) 
        print ("accuracy score: {0:.2f}%".format(acc_svc*100))
    return