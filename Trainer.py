"""
Created on Tue Oct 28 01:40:46 2022

@author: Md Ibrahim Khalil
"""


import cv2
from sklearn.svm import LinearSVC

class Trainer:
    def __init__(self, feature_extractor, train_data, test_data=None):
        self.feature_extractor = feature_extractor
        self.train_data = train_data
        self.test_data = test_data
        self.model = None

    def get_predictions(self, test_data):
        predictions  = []
        for item in test_data:
            prediction = self.model.predict(item.reshape(1, -1))    
            predictions.append(prediction[0])
        return predictions

    def train(self):
        X_train, Y_train, x_test, y_test = self.__get_feature_vectors()
        self.__train(X_train, Y_train)
        y_predicted = self.get_predictions(x_test)
        acc = self.__accuracy(y_test, y_predicted)
        return acc

    def __train(self, x_data, y_labels):
        self.model = LinearSVC(C=100.0, random_state=42)
        self.model.fit(x_data, y_labels)
    
    def __accuracy(self, y_test, y_predicted):
        cnt = 0
        for idx in range(len(y_predicted)):
            if y_test[idx]==y_predicted[idx]: cnt+=1
        return cnt/len(y_test)
    
    def __get_feature_vectors(self):
        X = []
        Y = []
        x = []
        y = []
        for clss in self.train_data.keys():
            for item in self.train_data[clss]:
                image = cv2.imread(item)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                hist = self.feature_extractor.describe(gray)
                X.append(hist)
                Y.append(clss)
            
            for item in self.test_data[clss]:
                image = cv2.imread(item)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                hist = self.feature_extractor.describe(gray)
                x.append(hist)
                y.append(clss)
        return X,Y,x,y # trainX,trainY,testx,testy

