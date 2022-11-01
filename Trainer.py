"""
Created on Tue Oct 28 01:40:46 2022

@author: Md Ibrahim Khalil
"""


import cv2
from sklearn.svm import LinearSVC, SVC
from skimage.io import imread
from skimage import data, color
from skimage.transform import rescale, resize
from skimage.filters import gaussian
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt



class Trainer:
    def __init__(self, feature_extractor, train_data, test_data=None):
        self.feature_extractor = feature_extractor
        self.train_data = train_data
        self.test_data = test_data
        self.model = None
        self.categories = ['Cloudy','Pen','Shiny']

    def get_predictions(self, test_data):
        predictions  = []
        for item in test_data:
            prediction = self.model.predict(item.reshape(1, -1))    
            predictions.append(prediction[0])
        return predictions

    def train(self):
        X_train, Y_train, x_test, y_test = self.__get_feature_vectors()
        self.__train(X_train, Y_train)
        # y_predicted = self.get_predictions(x_test)
        y_predicted = self.model.predict(x_test)
        acc = self.__accuracy(y_test, y_predicted)
        return acc

    def __train(self, x_data, y_labels):
        # self.model = LinearSVC(C=100.0, random_state=42)
        # self.model.fit(x_data, y_labels)
        param_grid = {'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
        svc = SVC(probability = True)
        self.model = GridSearchCV(svc, param_grid)
        self.model.fit(x_data, y_labels)
    
    def __accuracy(self, y_test, y_predicted):
        acc = {}
        for idx in range(len(y_predicted)):
            if y_test[idx] not in acc.keys():
                acc[y_test[idx]] = {}
                acc[y_test[idx]]["total"] = 0
                acc[y_test[idx]]["correct"] = 0
            acc[y_test[idx]]["total"] +=1
            if y_test[idx]==y_predicted[idx]: acc[y_test[idx]]["correct"] +=1
        cnt = 0
        for key in acc:
            cnt+=acc[key]['correct']
            print(key,": ",acc[key]['correct']/acc[key]['total'])
        print("overall",": ",cnt/len(y_test))
        self.show_confussion_mat(y_test, y_predicted)
        return acc
    
    def __preprocess(self, img, h=64, w=64):
        # print(img)
        image = imread(img)
        grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_array = gaussian(grayimage, sigma=0.4)
        img_resized = resize(img_array,(h, w), mode='constant', preserve_range=True)
        # print(img_resized)
        return img_resized


    def __get_feature_vectors(self):
        X = []
        Y = []
        x = []
        y = []
        for clss in self.train_data.keys():
            for item in self.train_data[clss]:
                img = self.__preprocess(item)
                hist = self.feature_extractor.describe(img)
                X.append(hist)
                Y.append(clss)
            
            for item in self.test_data[clss]:
                img = self.__preprocess(item)
                hist = self.feature_extractor.describe(img)
                x.append(hist)
                y.append(clss)
        return X,Y,x,y # trainX,trainY,testx,testy


    def show_confussion_mat(self, test_labels, y_pred):
        cm = confusion_matrix(test_labels, y_pred)
        print(cm)
        disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = self.categories)
        disp.plot()

