# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 02:13:41 2022

@author: arman hossain
"""
# !pip install scikit-image
# import the necessary packages
from skimage import feature
import numpy as np
import cv2
# from Lbp import LocalBinaryPatterns
from sklearn.svm import LinearSVC

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        
    # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
        
    def describe(self, image, eps=1e-7):
        
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints + 3), range=(0, self.numPoints + 2))
        
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        print(hist.shape, hist)
        # return the histogram of Local Binary Patterns
        return hist
    
    # def lbp_features(self,train,test):
    #     # desc = LocalBinaryPatterns(24, 8)
    #     X = []
    #     Y = []
    #     x= []
    #     y = []
    #     for clss in train.keys():
    #         for item in train[clss]:
    #             image = cv2.imread(item)
    #             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #             hist = self.describe(gray)
    #             X.append(hist)
    #             Y.append(clss)
            
    #         for item in test[clss]:
    #             image = cv2.imread(item)
    #             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #             hist = self.describe(gray)
    #             x.append(hist)
    #             y.append(clss)
    #     return X,Y,x,y # trainX,trainY,testx,testy

    # # https://github.com/shreyas-dharanesh/Fingerprint-Spoof-Detector-Based-on-Local-Binary-Patterns/blob/master/recognize.py

    # def lbp_train(self,x_data,y_labels):
    #     model = LinearSVC(C=100.0, random_state=42)
    #     model.fit(x_data, y_labels)
    #     return model

    # def get_predictions(self,lbp_model,test_data):
    #     predictions  = []
    #     for item in test_data:
    #         prediction = lbp_model.predict(item.reshape(1, -1))    
    #         predictions.append(prediction[0])
    #     return predictions

    # def accuracy(self,actualy,predictedy):
    #     cnt = 0
    #     for idx in range(len(predictedy)):
    #         if actualy[idx]==predictedy[idx]: cnt+=1
    #     return cnt/len(actualy)
    # def train_test(self,train,test):
    #     # files = get_all_files()
    #     # train,test = split_(files)
    #     X,Y,x,y = self.lbp_features(train, test)
    #     lbp_model = self.lbp_train(X,Y)
    #     predictedy = self.get_predictions(lbp_model, x)
    #     acc = self.accuracy(y, predictedy)
    #     return acc;