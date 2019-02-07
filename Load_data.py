# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 16:42:48 2019

@author: Ameya
"""

import glob
import errno
import pandas as pd
import numpy as np



train_pos = Read_Files('Pos','train/pos/*.txt',1)
train_neg = Read_Files('Neg','train/neg/*.txt',0)
test =  Read_Files('Test','test/*.txt',None)

def Read_Files(data_type,path,sentiment):
    data_type = []
    files = glob.glob(path)
    for name in files:
        try:
            with open(name, encoding="utf8") as f:
                data_type.append(f.readlines())
        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise
        framed_data = pd.DataFrame(data_type)
        framed_data['sentiment'] = sentiment
    return framed_data

train = pd.concat([train_pos,train_neg])




     

                

