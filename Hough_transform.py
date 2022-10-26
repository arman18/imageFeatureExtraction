"""
Created on Tue Oct 27 01:51:46 2022

@author: Md Ibrahim Khalil
"""


import cv2
import numpy as np


class HoughTransform:
    def __init__(self, image):
        self.img = image
        self.lines = None
        self.circles = None

    def get_lines(self):
        edges = cv2.Canny(gray, 50, 200)
        self.lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                                     100, minLineLength=10, maxLineGap=250)
        return self.lines

    def get_circles(self):
        img_blur = cv2.medianBlur(gray, 5)
        self.circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT,
                                        1, 40, param1=100, param2=30, minRadius=1, maxRadius=40)
        return self.circles

    def show_lines(self):
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(self.img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.imshow("Line Detected Image", self.img)
        cv2.waitKey(0)

    def show_circle(self):
        if self.circles is not None:
            circles = np.uint16(np.around(self.circles))
            for i in circles[0, :]:
                # Draw outer circle
                cv2.circle(self.img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw inner circle
                cv2.circle(self.img, (i[0], i[1]), 2, (0, 0, 255), 5)
        cv2.imshow('circles', self.img)
        cv2.waitKey(0)

    def show_detected_features(self):
        pass


if __name__ == '__main__':
    path = "C:/Users/user/Desktop/Ibrahim/Academic/pattern_recognition/Assignment-2/imageFeatureExtraction/data/class2/eyes.JPG"
    # path = "C:/Users/user/Desktop/Ibrahim/Academic/pattern_recognition/Assignment-2/imageFeatureExtraction/data/class2/1.png"
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ht = HoughTransform(gray)

    circles = ht.get_circles()
    ht.show_circle()

    lines = ht.get_lines()
    ht.show_lines()
