import cv2
from sklearn.svm import LinearSVC
from skimage.io import imread
from skimage import data, color
from skimage.transform import rescale, resize
from skimage.filters import gaussian




image = imread("./data/class1/cloudy66.jpg")
grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# grayimage = color.rgb2gray(image)
img_array = gaussian(grayimage, sigma=0.4)
img_resized = resize(img_array,(64, 64), mode='constant', preserve_range=True)
print(img_resized.shape)