from math import atan

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

imgHranyX = conv.convolve(img.astype(np.int16), Gx)
imgHranyY = conv.convolve(img.astype(np.int16), Gy)

edgeGradient = np.sqrt(np.power(imgHranyX, 2) + np.power(imgHranyY, 2))
edgeGradient[np.isnan(edgeGradient)] = 0

gAngle = np.atan(imgHranyY / imgHranyX)
gAngle[np.isnan(gAngle)] = 0

gDirection = np.round(gAngle * 8 / np.pi)
gDirection[gDirection < 0] = 4 + gDirection[gDirection < 0]
gDirection[gDirection == 4] = 0
gDirection = gDirection.astype(np.int8)

# print(np.min(edgeGradient).astype(np.uint8), np.max(edgeGradient).astype(np.uint8))
# print(np.min(gAngle), np.max(gAngle))
# print(np.min(gDirection), np.max(gDirection))
#
# cv.imshow('edgeGradient', edgeGradient)
# cv.imshow('gAndle', gAngle)
# cv.imshow('gDirection', gDirection /3*255)

cv.waitKey()


# todo Non-maximum Suppression

# todo Hysteresis Thresholding