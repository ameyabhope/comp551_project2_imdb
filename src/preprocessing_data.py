# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 13:01:26 2019

@author: Binary
"""
'''
This file has all the functions of preprocessing the raw text data by removing the noise, 
making the data lowercase, split the data, lemmatize the data and remove the stop words.
'''
import re    
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
stop_words = open("en_stopwords.txt", 'r' , encoding="ISO-8859-1").read()

#Progress bar
def process_data(data):
    data = data.progress_map(preprocess_data)  
    return data

#Preprocessing the data
def preprocess_data(review):
    document = remove_html(review)
    document = cleaning_data(document)
    document = lemmatize_data(document)
    document = stopwords_data(document)
    return document  
    
#Remove html tags
def remove_html(review):
    soup = BeautifulSoup(review, "html.parser")
    return soup.get_text()

# Removing the noise and negation handling            
def cleaning_data(document): 
    #Cleaning the data by removing special characters
    review = re.sub(r"[^A-Za-z0-9!?\'\`]", " ", document)
    #Handling negations
    review = re.sub(r"it's", " it is", review)
    review = re.sub(r"ain't", "is not",review)
    review = re.sub(r"aren't", "are not",review)
    review = re.sub(r"couldn't", "could not",review)
    review = re.sub(r"didn't", "did not",review)
    review = re.sub(r"doesn't", "does not",review)
    review = re.sub(r"hadn't", "had not",review)
    review = re.sub(r"hasn't", "has not",review)
    review = re.sub(r"haven't", "have not",review)
    review = re.sub(r"isn't", "is not",review)
    review = re.sub(r"shouldn't", "should not",review)
    review = re.sub(r"shan't", "shall not",review)
    review = re.sub(r"wasn't", "was not",review)
    review = re.sub(r"weren't", "were not",review)
    review = re.sub(r"oughtn't", "ought not",review)
    review = re.sub(r"that's", " that is", review)
    review = re.sub(r"\'s", " 's", review)
    review = re.sub(r"\'ve", " have", review)
    review = re.sub(r"won't", " will not", review)
    review = re.sub(r"wouldn't", " would not", review)
    review = re.sub(r"don't", " do not", review)
    review = re.sub(r"can't", " can not", review)
    review = re.sub(r"cannot", " can not", review)
    review = re.sub(r"n\'t", " n\'t", review)
    review = re.sub(r"\'re", " are", review)
    review = re.sub(r"\'d", " would", review)
    review = re.sub(r"\'ll", " will", review)
    review = re.sub(r"!", " ! ", review)
    review = re.sub(r"\?", " ? ", review)
    review = re.sub(r"\s{2,}", " ", review)
    # Removing all the numbers
    review = re.sub(r'[0-9]+', ' ', review) 
    #Removing all puncs
    review = re.sub(r'[^\w\s]','',review)
    # Substituting multiple spaces with single space
    review = re.sub(r'\s+', ' ', review, flags=re.I)
    #Lower case the data
    review = review.lower()
    return review

#Normalising the data by lemmatizing
def lemmatize_data(review):
    review = [lemmatizer.lemmatize(word) for word in review.split()]
    review = ' '.join(review)
    review = [word for word in review.split() if len(word) >= 3]
    review = ' '.join(review)
    return review
    
#Removing the stop words
def stopwords_data(review):
    review = [word for word in review.split() if not word in stop_words]
    review = ' '.join(review)           
    return review

#list of cleaned words
def cleandata(X_datatrain, X_datatest):
    cleandata_Train = []
    for sen in range(0, len(X_datatrain)): 
        cleandata_Train.append(preprocess_data(str(X_datatrain[sen])))
        
    cleandata_Test = []
    for sen in range(0, len(X_datatest)): 
        cleandata_Test.append(preprocess_data(str(X_datatest[sen])))
    return cleandata_Train, cleandata_Test
