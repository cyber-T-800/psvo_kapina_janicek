from random import gauss

import numpy as np
import cv2 as cv
import convolution as conv

image = cv.imread('messi.png', cv.IMREAD_GRAYSCALE)
image = cv.resize(image, (400, 400))

gauss = np.array([
    [0.002969, 0.013306, 0.021938, 0.013306, 0.002969],
    [0.013306, 0.059634, 0.09832,  0.059634, 0.013306],
    [0.021938, 0.09832,  0.1621,   0.09832,  0.021938],
    [0.013306, 0.059634, 0.09832,  0.059634, 0.013306],
    [0.002969, 0.013306, 0.021938, 0.013306, 0.002969]
])
# todo noise reduction by gaussian 5x5 filter

img = conv.convolve(image, gauss)


cv.imshow('image', image)
cv.imshow('img', img)

cv.waitKey()

# todo Finding Intensity Gradient of the Image

# todo Non-maximum Suppression

# todo Hysteresis Thresholding