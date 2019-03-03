# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 13:03:41 2019

@author: Binary
"""
'''
This file provides us the function for vectorizing the data and splitting the Cleandata_Train
 for Training and validation
 Vectorization involves:
     1. Count vectorisation
     2. tfidf transformer
'''
#Importing the libraries
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.model_selection import train_test_split  


#Vectorisation (Count vectorisation with tfidf transformer)
def vectorization(cleandata_Train, cleandata_Test):
    vectorizer = CountVectorizer(analyzer='word', min_df=2,max_df= 0.9, ngram_range=(1, 3))  
    X_train_vect = vectorizer.fit_transform(cleandata_Train) 
    X_test_vect = vectorizer.transform(cleandata_Test) 
    
    tfidfconverter = TfidfTransformer(sublinear_tf=True, use_idf =True, norm='l2')  
    X_train_tfidf = tfidfconverter.fit_transform(X_train_vect)
    X_test_tfidf = tfidfconverter.transform(X_test_vect)
    return  X_train_tfidf, X_test_tfidf

#Spliting the dataset for training and validation
def traintestsplit(X_train_tfidf, X_test_tfidf, y_datatrain):
    X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, y_datatrain, test_size=0.17, random_state=42) 
    return X_train, X_test, y_train, y_test

#Vectorisation (Count vectorisation with tfidf transformer) for Bernoullis Naive Bayes and NBSVM
def vectorization_nb(cleandata_Train, cleandata_Test):
    vectorizer = CountVectorizer(max_features= 10000, min_df=2, max_df= 0.8, ngram_range=(1, 3))  
    X_train_vect = vectorizer.fit_transform(cleandata_Train) 
    X_test_vect = vectorizer.transform(cleandata_Test) 
    
    tfidfconverter = TfidfTransformer(sublinear_tf=True, use_idf =True, norm='l2')  
    X_train_tfidf = tfidfconverter.fit_transform(X_train_vect)
    X_test_tfidf = tfidfconverter.transform(X_test_vect)
    return  X_train_tfidf, X_test_tfidf

