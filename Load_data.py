import glob
import errno
import pandas as pd
import numpy as np
import re


train_pos = Read_Files('Pos','train/pos/*.txt',1)
train_neg = Read_Files('Neg','train/neg/*.txt',0)
test =  Read_Files('Test','test/*.txt',None)

def Read_Files(data_type,path,sentiment):
    data_type = []
    files = glob.glob(path)
    for name in files:
            with open(name, encoding="utf8") as f:
                data_type.append(f.readlines())
                framed_data = pd.DataFrame(data_type)
                framed_data['sentiment'] = sentiment
    return framed_data

train = pd.concat([train_pos,train_neg])

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train.iloc[:,0], train.iloc[:,1], test_size = 0.2, random_state =10)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer().fit(X_train)
X_train_counts = count_vect.transform(X_train)
X_val_counts = count_vect.transform(X_val)
X_test_counts = count_vect.transform(test.iloc[:,0])


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_tfidf = tfidf_transformer.transform(X_train_counts)
X_val_tfidf = tfidf_transformer.transform(X_val_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

from sklearn.preprocessing import Normalizer

normalizer_tranformer = Normalizer().fit(X=X_train_tfidf)
X_train_normalized = normalizer_tranformer.transform(X_train_tfidf)
X_val_normalized = normalizer_tranformer.transform(X_val_tfidf)
X_test_normalized = normalizer_tranformer.transform(X_test_tfidf)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
for c in [0.01, 0.05, 0.25, 0.5, 1]:
        lr = LogisticRegression()
        lr.fit(X_train_normalized, y_train)
        print ("Accuracy for C=%s: %s" 
           % (c, accuracy_score(y_test, lr.predict(X_val_normalized))))

from sklearn.svm import LinearSVC
for c in [0.01, 0.05, 0.25, 0.5, 1]:
        svc = LinearSVC()
        svc.fit(X_train_normalized, y_train)
        print ("Accuracy for C=%s: %s" 
           % (c, accuracy_score(y_test, svc.predict(X_val_normalized))))      

from sklearn.naive_bayes import GaussianNB
for c in [0.01, 0.05, 0.25, 0.5, 1]:
        gnb = GaussianNB()
        gnb.fit(X_train_normalized, y_train)
        print ("Accuracy for C=%s: %s" 
           % (c, accuracy_score(y_test, gnb.predict(X_val_normalized))))  

a = []
a = pd.DataFrame(clf.predict(X_test_normalized))

a.to_csv('Sub.csv',index=True)
