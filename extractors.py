import cv2
import numpy as np
from math import copysign, log10
from skimage import feature

# Os extratores mais comuns são:
# HU, LBP e GLCM

class Extractors:

    def huMoments(self, X):

        X_new = []
        for img in X:
            # Convert array in image
            image = img.astype(np.uint8)

            # Calculate Moments
            moments = cv2.moments(image)

            # Calculate Hu Moments
            huMoments = cv2.HuMoments(moments)

            # Log scale hu moments
            hm = [-1 * copysign(1.0, hu) * log10(abs(hu)) if hu != 0 else 0 for hu in huMoments]

            X_new.append(hm)

        return np.array(X_new)

    def lbp(self, X, eps=1e-7):

        X_new = []
        for img in X:
            # Convert array in image
            image = img.astype(np.uint8)

            # Calculate patterns
            lbp = feature.local_binary_pattern(image, 24, 8, method="uniform")
            (hist, _) = np.histogram(lbp.ravel(),
                                     bins=np.arange(0, 24 + 3),
                                     range=(0, 24 + 2))

            # normalize the histogram
            hist = hist.astype("float")
            hist /= (hist.sum() + eps)

            # return the histogram of Local Binary Patterns
            X_new.append(hist)

        return np.array(X_new)

    def glcm(self, X):

        X_new = []
        for img in X:
            g = feature.greycomatrix(np.array(img, dtype='uint8'), [1, 2], [0, np.pi / 4, np.pi / 2], 2, normed=True, symmetric=True)

            contrast = feature.greycoprops(g, 'contrast').tolist()
            dissimilarity = feature.greycoprops(g, 'dissimilarity').tolist()
            homogeneity = feature.greycoprops(g, 'homogeneity').tolist()
            asm = feature.greycoprops(g, 'ASM').tolist()
            energy = feature.greycoprops(g, 'energy').tolist()
            correlation = feature.greycoprops(g, 'correlation').tolist()

            values = contrast[0] + contrast[1] + dissimilarity[0] + dissimilarity[1] + homogeneity[0] + \
                      homogeneity[1] + asm[0] + asm[1] + energy[0] + energy[1] + correlation[0] + correlation[1]

            X_new.append(values)

        return X_new
