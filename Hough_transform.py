"""
Created on Tue Oct 27 01:51:46 2022

@author: Md Ibrahim Khalil
"""


import cv2
import numpy as np
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line



class HoughTransform:
    def __init__(self):
        self.img = None
        self.lines = None
        self.circles = None

    def get_lines(self, img, show=False):
        # edges = cv2.Canny(img, 50, 200)
        # self.lines = cv2.HoughLinesP(img, 1, np.pi/180,
        #                              100, minLineLength=10, maxLineGap=250)
        # self.lines = self.lines if self.lines is not None else np.zeros((1,1,1))

        edges = canny(img, 2, 1, 100)
        self.lines = probabilistic_hough_line(edges, threshold=10, line_length=5,
                                        line_gap=3)
        self.lines = np.array(self.lines)
        if show:
            self.__show_lines()
        return self.lines

    def get_circles(self, img, show=False):
        img_blur = cv2.medianBlur(img, 5)
        self.circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT,
                                        1, 40, param1=100, param2=30, minRadius=1, maxRadius=40)
        self.circles = self.circles if self.circles is not None else np.zeros((1,1,1))
        if show:
            self.__show_circle()
        return self.circles.flatten()

    def describe(self, img):
        self.img = img
        lines = self.get_lines(img)
        # circles = self.get_circles(img)
        # ft = np.hstack((lines, circles))
        # n =ft.shape[0]
        # if n >= 256:
        #     ft = ft[:256]
        # else:
        #     ft = np.hstack((ft, np.zeros(256-n)))
        n = lines.shape[0]
        if n >= 256:
            lines = lines[:256]
        else:
            lines = np.hstack((lines, np.zeros(256-n)))
        return lines

    def __show_lines(self):
        for line in self.lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(self.img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.imshow("Line Detected Image", self.img)
        cv2.waitKey(0)

    def __show_circle(self):
        if self.circles is not None:
            circles = np.uint16(np.around(self.circles))
            for i in circles[0, :]:
                # Draw outer circle
                cv2.circle(self.img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw inner circle
                cv2.circle(self.img, (i[0], i[1]), 2, (0, 0, 255), 5)
        cv2.imshow('circles', self.img)
        cv2.waitKey(0)


if __name__ == '__main__':
    path = "C:/Users/bs107/Desktop/imageFeatureExtraction/data_train/cat/cat_0077.jpg"
    # path = "C:/Users/user/Desktop/Ibrahim/Academic/pattern_recognition/Assignment-2/imageFeatureExtraction/data/class2/1.png"
    # img = cv2.imread(path, cv2.IMREAD_COLOR)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ht = HoughTransform()

    # circles = ht.get_circles(gray, True)
    # ht.show_circle()

    # lines = ht.get_lines()
    # ht.show_lines()

    # feature = ht.describe(gray)

    # print(feature.shape, feature)
