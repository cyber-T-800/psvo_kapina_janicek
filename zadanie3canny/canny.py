from random import gauss

import numpy as np
import cv2 as cv
import convolution as conv

image = cv.imread('messi.png', cv.IMREAD_GRAYSCALE)
image = cv.resize(image, (400, 400))

#gaus
gauss = np.array([
    [0.002969, 0.013306, 0.021938, 0.013306, 0.002969],
    [0.013306, 0.059634, 0.09832,  0.059634, 0.013306],
    [0.021938, 0.09832,  0.1621,   0.09832,  0.021938],
    [0.013306, 0.059634, 0.09832,  0.059634, 0.013306],
    [0.002969, 0.013306, 0.021938, 0.013306, 0.002969]
])

img = conv.convolve(image, gauss)


# gradient
Gx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])

Gy = np.array([[ 1,  2,  1],
               [ 0,  0,  0],
               [-1, -2, -1]])

imgHranyX = conv.convolve(img, Gx)
imgHranyY = conv.convolve(img, Gy)


cv.imshow('imageX', imgHranyX)
cv.imshow('imageY', imgHranyY)
cv.waitKey()

# todo Non-maximum Suppression

# todo Hysteresis Thresholding