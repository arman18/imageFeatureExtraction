# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 00:44:07 2022

@author: arman hossain
"""
import os
import numpy as np
import cv2

def get_all_files(path="data"): # assumed one level hierarchy
    '''
    {
     'class1': ['data\\class1\\2.png', 'data\\class1\\1.png'],
     'class2': ['data\\class2\\1.png'],
     'class3': ['data\\class3\\1.png']
     }
    '''    
    dictionary = {}
    for root, dirs, files in os.walk(path):
        if root!=path: dictionary[root.split("\\")[-1]] = []
        
        for filename in files:
            file = os.path.join(root, filename)
            dictionary[root.split("\\")[-1]].append(file)
    # print(dictionary)       
    return dictionary
        
def split_(files, factors = 0.2):
    train = {}
    testing = {}
    
    for clss in files.keys():
        testing[clss] = np.random.choice(files[clss], int(len(files[clss])*factors), replace=False)
        train[clss] = []
        for item in files[clss]:
            if item in testing[clss]: continue
            train[clss].append(item)
    return train,testing