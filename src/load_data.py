# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 13:05:37 2019

@author: Binary
"""
'''
This file has function to read the textfiles from training and testing data folders 
'''

# Importing the libraries
import glob
import pandas as pd

#Reading the training data files
def Read_Files_train(data_type,path):
    data_type = []
    files = sorted(glob.glob(path))
    for name in files:
            with open(name, encoding="utf8") as f:
                data_type.append(f.readlines())
    return data_type

#Reading the testing data files by preserving the index
def Read_Files_test(data_type,path):
    k = []
    index = []
    data_type = []
    files = sorted(glob.glob(path))
    for name in files:
            with open(name, encoding="utf8") as f:
                index  = name
                k.append(index[index.find('s')+3:index.find('.')])
                data_type.append(f.readlines())
    framed_data = pd.DataFrame(data_type)
    framed_data.columns = ['comment']
    framed_data['order'] = k
    del k, index
    return framed_data 

