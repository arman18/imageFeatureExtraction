from skimage.feature import hog
import numpy as np

class HOG:
    def __init__(self, ):
        self.pixels_per_cell = (16, 16)
        
    def describe(self, image):
        
        hog_features, hog_image = hog(image,
                              visualize=True,
                              block_norm='L2-Hys',
                              pixels_per_cell=self.pixels_per_cell)

        n =hog_features.shape[0]
        if n >= 256:
            hog_features = hog_features[:256]
        else:
            hog_features = np.hstack((hog_features, np.zeros(256-n)))
        return hog_features
        return hog_features