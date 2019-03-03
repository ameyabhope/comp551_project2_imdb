"""
Created on Mon Feb 18 13:21:31 2019

@author: Binary
"""
'''
This file has the main function where the cleaning, vectorization and classification is 
implemented as a model.
            "Check the Jupyter notebook file for graphs, wordclouds and outputs."
'''
import pandas as pd
from sklearn.metrics import accuracy_score
from load_data import Read_Files_train, Read_Files_test
from preprocessing_data import process_data
from vectorization_split_data import vectorization, traintestsplit, vectorization_nb
from classifiers import logisticreg, ridgeclassifier, decisiontree, linearSVC, MNB, nbsvm, stochastic_GD
from pipeline_data import pipeline_data
from naive_bayes_from_scratch import BernoulliNB

def main():
    '''
    Reading the data from pos and neg folder for training and test folder for testing 
    The directory of the folders should match
    '''
    print("Reading of the data files on process...")
    train_pos = pd.DataFrame(Read_Files_train('Pos','../data/train/pos/*.txt'))
    train_pos.columns = ['comment']
    train_pos['pos_neg'] = 1
    train_neg = pd.DataFrame(Read_Files_train('Neg','../data/train/neg/*.txt'))
    train_neg.columns = ['comment']
    train_neg['pos_neg'] = 0
    X_datatrain_posneg = pd.concat([train_pos,train_neg])
    test =  Read_Files_test('Test','../data/test/*.txt')
    test['order'] = pd.to_numeric(test['order'])
    test = test.sort_values('order', ascending=True)
    test = test.reset_index()
    X_datatrain = X_datatrain_posneg['comment']
    y_datatrain = X_datatrain_posneg['pos_neg']
    X_datatest = test['comment']    
    print("Task Completed")
    

    #Save predicted value in csv
    def save2csv(y_pred_test):
        rawdata= { 'Category': y_pred_test }
        a = pd.DataFrame(rawdata, columns = ['Category'])
        return a.to_csv('yPred.csv',index=True, header=True)

    #Accuracy, confusion matrix
    def confusion_matrix(ypred, y_test):
        from sklearn.metrics import classification_report, confusion_matrix
        print(confusion_matrix(y_test , ypred))  
        print(classification_report(y_test,ypred))  
        print(accuracy_score(y_test, ypred)) 
        Accuracy = accuracy_score(y_test, ypred)
        return Accuracy
    
    '''
    Cleaning of the data is done    
    '''
    print("Cleaning of Training data on process...")
    cleandata_Train = process_data(X_datatrain)
    print("Task completed")
    print("Cleaning of Testing data on process...")
    cleandata_Test = process_data(X_datatest)
    print("Task completed")
   
    print("Press \n '1' for Naive Bayes from scratch and regular Classifiers \n '2' for Model Valiation Pipeline (with binary occurances vs TfIdf weighting)"  )
    a= input("Enter the selection: ")
    a = int(a)
    if a==1:
        ''' List of classifiers,
            choose your desired classifier:
            1. Bernoulli Naive Bayes from scratch (Keep a check on the memory, maxures = 25000 in vectorisation to avoid MEMORY ERROR)
            2. Logistic 
            3. Linear SVC (Our Best model)
            4. Multinominal Naive Bayes
            5. Ridge Classifier
            6. Stochastic Gradient Descent
            7. Naive Bayes SVM _feat(Keep a check on the memory, maxures = 25000 in vectorisation to avoid MEMORY ERROR)
            8. Decision tree
        '''
        print("Press \n '1' for Bernoulli Naive Bayes from scratch  \n '2' for Logistic regression.\n '3' for Linear SVC. (Our Best model) \n '4' for Multinominal Naive Bayes.  \n '5' for Stochastic Gradient Descent \n '6' Ridge classifier.  \n '7' for Naive Bayes SVM.  \n '8' for Decision tree." )
        b= input("Enter the selection: ")
        b = int(b)
        if b==1:
            print("Vectorization on process...")
            X_train_tfidf, X_test_tfidf = vectorization_nb(cleandata_Train, cleandata_Test)
            print("Task completed")
            print("Splitting of Training data for training and validation on process")
            X_train, X_test, y_train, y_test = traintestsplit(X_train_tfidf, X_test_tfidf, y_datatrain)
            print("Task completed")            
            print("Performing Naive Bayes from Scratch...")
            ypred, y_pred_test   = BernoulliNB(X_train, y_train, X_test, y_test, X_test_tfidf)
            print("Task completed")                   
        elif b==2:
            print("Vectorization on process...")
            X_train_tfidf, X_test_tfidf = vectorization(cleandata_Train, cleandata_Test)
            print("Task completed")
            print("Splitting of Training data for training and validation on process")
            X_train, X_test, y_train, y_test = traintestsplit(X_train_tfidf, X_test_tfidf, y_datatrain)
            print("Task completed")
            print("Performing Logisitc regression...")
            ypred, y_pred_test   = logisticreg(X_train, y_train, X_test, y_test, X_test_tfidf)
            print("Task completed")
        elif b==3:
            print("Vectorization on process...")
            X_train_tfidf, X_test_tfidf = vectorization(cleandata_Train, cleandata_Test)
            print("Task completed")
            print("Splitting of Training data for training and validation on process")
            X_train, X_test, y_train, y_test = traintestsplit(X_train_tfidf, X_test_tfidf, y_datatrain)
            print("Task completed")            
            print("Performing Linear SVC...")
            ypred, y_pred_test   = linearSVC(X_train, y_train, X_test, y_test, X_test_tfidf)
            print("Task completed")
        elif b==4:
            print("Vectorization on process...")
            X_train_tfidf, X_test_tfidf = vectorization(cleandata_Train, cleandata_Test)
            print("Task completed")
            print("Splitting of Training data for training and validation on process")
            X_train, X_test, y_train, y_test = traintestsplit(X_train_tfidf, X_test_tfidf, y_datatrain)
            print("Task completed")            
            print("Performing Multinominal Naive Bayes...")
            ypred, y_pred_test   = MNB(X_train, y_train, X_test, y_test, X_test_tfidf)
            print("Task completed")
        elif b==5:
            print("Vectorization on process...")
            X_train_tfidf, X_test_tfidf = vectorization(cleandata_Train, cleandata_Test)
            print("Task completed")
            print("Splitting of Training data for training and validation on process")
            X_train, X_test, y_train, y_test = traintestsplit(X_train_tfidf, X_test_tfidf, y_datatrain)
            print("Task completed")                        
            print("Performing Stochastic Gradient Descent...")
            ypred, y_pred_test   = stochastic_GD(X_train, y_train, X_test, y_test, X_test_tfidf)
            print("Task completed")
        elif b==6:
            print("Vectorization on process...")
            X_train_tfidf, X_test_tfidf = vectorization(cleandata_Train, cleandata_Test)
            print("Task completed")
            print("Splitting of Training data for training and validation on process")
            X_train, X_test, y_train, y_test = traintestsplit(X_train_tfidf, X_test_tfidf, y_datatrain)
            print("Task completed")            
            print("Performing Ridge Classification...")
            ypred, y_pred_test    = ridgeclassifier(X_train, y_train, X_test, y_test, X_test_tfidf)
            print("Task completed")
        elif b==7:
            print("Vectorization on process...")
            X_train_tfidf, X_test_tfidf = vectorization_nb(cleandata_Train, cleandata_Test)
            print("Task completed")
            print("Splitting of Training data for training and validation on process")
            X_train, X_test, y_train, y_test = traintestsplit(X_train_tfidf, X_test_tfidf, y_datatrain)
            print("Task completed")            
            print("Performing Naive Bayes SVM...")
            ypred, y_pred_test   = nbsvm(X_train, y_train, X_test, y_test, X_test_tfidf)   
            print("Task completed")
        elif b==8:
            print("Vectorization on process...")
            X_train_tfidf, X_test_tfidf = vectorization(cleandata_Train, cleandata_Test)
            print("Task completed")
            print("Splitting of Training data for training and validation on process")
            X_train, X_test, y_train, y_test = traintestsplit(X_train_tfidf, X_test_tfidf, y_datatrain)
            print("Task completed")            
            print("Performing Decision tree...")
            ypred, y_pred_test   = decisiontree(X_train, y_train, X_test, y_test, X_test_tfidf)
            print("Task completed")
        else:
            print("Choose a valid option")
        
        '''
        printing accuracy and confusion matrix
        '''
        confusion_matrix(ypred, y_test)
        '''
        predicted values saved to csv
        '''
        save2csv(y_pred_test) 
        
    elif a==2:
        '''Pipelining of the data
            Pipelining is done considering 
            1. Binary Count vectorisation
            2. Count vectorisation with TfIdf 
            3. With Linear SVC
            '''
        print("Pipelining of data on-process...")
        pipeline_data(cleandata_Train, y_datatrain)
    else:
        print("Choose a valid option")
        

if __name__ == "__main__":
    main()