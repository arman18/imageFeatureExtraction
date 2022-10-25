# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 09:48:17 2022

@author: arman hossain
"""
from data_utils import *
from Lbp import LocalBinaryPatterns



files = get_all_files()
train,test = split_(files)
lbp = LocalBinaryPatterns(24, 8)
acc = lbp.train_test(train, test)
