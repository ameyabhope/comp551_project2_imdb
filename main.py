# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 18:01:32 2019

@author: Binary
"""
#importing the  libraries
import pandas as pd
import numpy as np  
import re  
import nltk  
nltk.download('stopwords')  
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.model_selection import train_test_split  
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
import warnings
warnings.filterwarnings("ignore")

#Reading the data
data_train = pd.read_csv(r'D:\Mcgill\Winter 2019\Applied machine learning\Testrun2\qwerty2\aclImdb_v1\aclImdb\imdb_train_trail.csv')
data_test  = pd.read_csv(r'D:\Mcgill\Winter 2019\Applied machine learning\Testrun2\qwerty2\aclImdb_v1\aclImdb\imdb_test_trial.csv')
X_datatrain=data_train["comment"]
y_datatrain=data_train["pos_neg"]
X_datatest = data_test["comment"]
y_datatest = data_test["pos_neg"]

#Preprocessing the data
def preprocess_data(review):
        document = cleaning_data(review)
        document = lowercase_split_data(document)
        document = lemmatize_data(document)
        document = stopwords_data(document)
        return document
    
#Cleaning the data by removing special characters, numbers, tags, punctuations              
def cleaning_data(review): 
        # Removing all the special characters
        review = re.sub(r'\W', ' ', review)
        review = re.sub(r'https?:\/\/.*\/[a-zA-Z0-9]*', ' ', review)    
        # Removing all the tags
        review = re.sub(r'<[^<>]+>', " ", review) 
        # Removing all the numbers
        review = re.sub(r'[0-9]+', ' ', review) 
        #Removing all puncs
        review = re.sub(r'[^\w\s]','',review)
        #Removing all the single characters
        review = re.sub(r'\s+[a-zA-Z]\s+', ' ', review)
        # Substituting multiple spaces with single space
        review = re.sub(r'\s+', ' ', review, flags=re.I)
        # Removing prefixed 'b'
        review = re.sub(r'^b\s+', '', review)
        return review
    
#Lower case and splitting the data   
def lowercase_split_data(review):
        # Converting to Lowercase
        review = review.lower()
        #Splitting the data
        review = review.split()
        return review
    
#Normalising the data by lemmatizing
def lemmatize_data(review):
        # Lemmatization of the data
        review = [lemmatizer.lemmatize(word) for word in review]
        review = ' '.join(review)
        review = [word for word in review.split() if len(word) > 2]
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

#Vectorisation (bigram with tfidf)
def vectorization(cleandata_Train, cleandata_Test):
    vectorizer = CountVectorizer(max_features=20000,encoding='latin-1',min_df=5, max_df=0.75, ngram_range=(1, 2))  
    X_train_vect = vectorizer.fit_transform(cleandata_Train) 
    X_test_vect = vectorizer.transform(cleandata_Test) 
    
    tfidfconverter = TfidfTransformer(norm='l2')  
    X_train_tfidf = tfidfconverter.fit_transform(X_train_vect)
    X_test_tfidf = tfidfconverter.transform(X_test_vect)
    return  X_train_tfidf, X_test_tfidf

#Spliting the dataset for training and validation
def traintestsplit(X_train_tfidf, X_test_tfidf, y_datatrain):
    X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, y_datatrain, test_size=0.2, random_state=101) 
    return X_train, X_test, y_train, y_test

#Classifiers
#Logisitic Regression
def logisticreg(X_train, y_train, X_test, y_test, X_test_tfidf):
    param_grid = {'C': [2, 2.5, 2.9 , 3, 3.1,  4, 4.5, 5 , 5.5, 6, 6.5]}
    grid = GridSearchCV(LogisticRegression(solver='lbfgs',multi_class='multinomial',random_state=0), param_grid, cv=10)
    grid.fit(X_train, y_train)
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))
    print("Best parameters: ", grid.best_params_)
    print("Best estimator: ", grid.best_estimator_)
    lr = grid.best_estimator_
    acc_lr=accuracy_score(y_test, lr.predict(X_test)) 
    y_pred_log_cv=lr.predict(X_test_tfidf)
    return acc_lr, y_pred_log_cv

#RidgeClassifier
def ridgeclassifier(X_train, y_train, X_test, y_test, X_test_tfidf): 
    from sklearn.linear_model import RidgeClassifier
    rdc_clf = RidgeClassifier(alpha =4, class_weight = 'balanced')
    rdc_clf.fit(X_train, y_train)
    acc_rd=accuracy_score(y_test, rdc_clf.predict(X_test)) 
    y_pred_log_rdc=rdc_clf.predict(X_test_tfidf)
    return acc_rd, y_pred_log_rdc

#Decision tree
def Decisiontree(X_train, y_train, X_test, y_test, X_test_tfidf):
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100)
    clf_entropy.fit(X_train, y_train)
    acc_dr=accuracy_score(y_test, clf_entropy.predict(X_test)) 
    y_pred_log_dt=clf_entropy.predict(X_test_tfidf)
    return acc_dr, y_pred_log_dt

#Support Vector Machine
def linearSVC(X_train, y_train, X_test, y_test, X_test_tfidf):
    Cs = np.logspace(-6, -1, 10)
    svc_clf = GridSearchCV(LinearSVC(), param_grid=dict(C=Cs), cv=5)
    svc_clf.fit(X_train, y_train)
    print("Best cross-validation score: {:.2f}".format(svc_clf.best_score_))
    print("Best parameters: ", svc_clf.best_params_)
    print("Best estimator: ", svc_clf.best_estimator_)
    svc_clf = svc_clf.best_estimator_
    acc_svm=accuracy_score(y_test, svc_clf.predict(X_test)) 
    y_pred_log_svm=svc_clf.predict(X_test_tfidf)
    return acc_svm, y_pred_log_svm

#Multinomial Naive Bayes model
def MNB(X_train, y_train, X_test, y_test, X_test_tfidf):
    nb_clf = MultinomialNB(alpha=1.5)
    nb_clf.fit(X_train, y_train)
    acc_MNB=accuracy_score(y_test, nb_clf.predict(X_test)) 
    y_pred_nb=nb_clf.predict(X_test_tfidf)
    return acc_MNB, y_pred_nb

#Naive Bayes Support vector machine
#Beware of the memory
def nbsvm(X_train, y_train, X_test, y_test):
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
    acc_Nbsvm=accuracy_score(y_test, mnbsvm.predict(X_test.toarray())) 
    y_pred_nbsvm=mnbsvm.predict(X_test_tfidf)

    return acc_Nbsvm, y_pred_nbsvm

#Save predicted value in csv
def save2csv(ypred):
    rawdata= { 'LogisticReg': ypred }
    a = pd.DataFrame(rawdata, columns = ['category'])
    return a.to_csv('yPred.csv',index=True, header=True)

##Accuracy, confusion matrix
def confusion_matrix(y_datatest, ypred):
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_datatest,ypred))  
    print(classification_report(y_datatest,ypred))  
    print(accuracy_score(y_datatest, ypred)) 
    return

#Pipelining model
def pipeline_data(cleandata_Train, y_datatrain):
    x_train, x_validation, y_train, y_validation = train_test_split(cleandata_Train, y_datatrain, test_size=.2, random_state=2000)
    
    from sklearn.pipeline import Pipeline
    stop_words = set(stopwords.words('english'))
    # build the pipeline
    ppl = Pipeline([
                  ('vect', CountVectorizer()),
                  ('tfidf', TfidfTransformer()),
                  ('clf',   LogisticRegression())
          ])
    
    params = {"vect__ngram_range": [(1,2)],
              "vect__binary" : [True, False],                     
              "vect__max_df": [0.8],
              "vect__stop_words" : [stop_words],
              "vect__min_df" : [5],
              "vect__max_features": [3000],
              "tfidf__use_idf": [True],
              "clf__C" : [3, 4]}
    grid = GridSearchCV(ppl, param_grid= params, n_jobs= -1, cv=10)
    grid.fit(x_train, y_train)
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))
    print("Best parameters: ", grid.best_params_)
    print("Best estimator: ", grid.best_estimator_)
    lr = grid.best_estimator_
    acc_lr=accuracy_score(y_validation, lr.predict(x_validation)) 
    print ("accuracy score: {0:.2f}%".format(acc_lr*100))
    return acc_lr

#Main program
#Cleaning of the data is done    
cleandata_Train, cleandata_Test = cleandata(X_datatrain, X_datatest)
print("Press '1' for regular method and '2' for pipelining" )
a= input("Enter the selection: ")
if (a==1):
    #Vectorisation of training and testing data
    X_train_tfidf, X_test_tfidf = vectorization(cleandata_Train, cleandata_Test)
    #Splitting of data train for training and validation
    X_train, X_test, y_train, y_test = traintestsplit(X_train_tfidf, X_test_tfidf, y_datatrain)
    #List of classifiers, choose your desired classifier
    acc_lr, y_pred_log_cv      = logisticreg(X_train, y_train, X_test, y_test, X_test_tfidf)
#    acc_rf, y_pred_log_rfc    = ridgeclassifier(X_train, y_train, X_test, y_test, X_test_tfidf)
#    acc_dr, y_pred_log_dt     = Decisiontree(X_train, y_train, X_test, y_test, X_test_tfidf)
#    acc_svm, y_pred_log_svm   = linearSVC(X_train, y_train, X_test, y_test, X_test_tfidf)
#    acc_MNB, y_pred_nb        = MNB(X_train, y_train, X_test, y_test, X_test_tfidf)   
#    acc_Nbsvm, y_pred_nbsvm   = nbsvm(X_train, y_train, X_test, y_test, X_test_tfidf)
    #Confusion matrix is created
    confusion_matrix(y_datatest, y_pred_log_cv) # Change ypred values according to the classifier
    #predicted values saved to csv
    save2csv(y_pred_log_cv) ## Change ypred values according to the classifier
elif (a==2):
    #Pipelining of the data
    pipeline_data(cleandata_Train, y_datatrain)
    



