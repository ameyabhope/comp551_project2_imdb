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
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,0], train.iloc[:,1], test_size = 0.2, random_state =10)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer().fit(X_train)
X_train_counts = count_vect.transform(X_train)
X_test_counts = count_vect.transform(X_test)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_tfidf = tfidf_transformer.transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

from sklearn.preprocessing import Normalizer

normalizer_tranformer = Normalizer().fit(X=X_train_tfidf)
X_train_normalized = normalizer_tranformer.transform(X_train_tfidf)
X_test_normalized = normalizer_tranformer.transform(X_test_tfidf)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_normalized, y_train)


from sklearn import metrics
y_pred = clf.predict(X_test_normalized)
print(metrics.classification_report(y_test, y_pred,
    target_names=newsgroups.target_names))