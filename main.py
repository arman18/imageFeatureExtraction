# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 09:48:17 2022

@author: arman hossain
"""
from data_utils import *
from Lbp import LocalBinaryPatterns
from Hough_transform import HoughTransform
from Trainer import Trainer
from Hog import HOG

files = get_all_files()
train, test = split_(files)

# print(len(test['class2']))


print("lbp feature_extractor")
lbp = LocalBinaryPatterns(24, 8)
lbp_trainer = Trainer(lbp, train, test)
print("****************SVM****************")
acc = lbp_trainer.train()
print("****************RandomForest****************")
rf_acc = lbp_trainer.train_rf()


print("HoughTransform feature_extractor")
ht = HoughTransform()
ht_trainer = Trainer(ht, train, test)
print("****************SVM****************")
acc = ht_trainer.train()
print("****************RandomForest****************")
rf_acc = lbp_trainer.train_rf()



print("HOG feature_extractor")
hog = HOG()
hog_trainer = Trainer(hog, train, test)
print("****************SVM****************")
acc = hog_trainer.train()
print("****************RandomForest****************")
rf_acc = lbp_trainer.train_rf()


if __name__ == "__main__":
    pass