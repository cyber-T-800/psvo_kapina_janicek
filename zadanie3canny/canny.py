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

def non_max_suppression(magnitude, angle_matrix):
    M, N = magnitude.shape
    Z = np.zeros((M, N), dtype=np.float32)

# todo Hysteresis Thresholding
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q = 255
            r = 255

            if (0 <= angle_matrix[i, j] < 22.5) or (157.5 <= angle_matrix[i, j] <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif 22.5 <= angle_matrix[i, j] < 67.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            elif 67.5 <= angle_matrix[i, j] < 112.5:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            elif 112.5 <= angle_matrix[i, j] < 157.5:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                Z[i, j] = magnitude[i, j]
            else:
                Z[i, j] = 0
    return Z


img_nms = non_max_suppression(edgeGradient, gAngle)

# todo Double thresholding
def double_threshold(img, low_ratio=0.05, high_ratio=0.15):
    high_thresh = img.max() * high_ratio
    low_thresh = high_thresh * low_ratio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.uint8)

    weak = 25
    strong = 255

    strong_i, strong_j = np.where(img >= high_thresh)
    zeros_i, zeros_j = np.where(img < low_thresh)
    weak_i, weak_j = np.where((img <= high_thresh) & (img >= low_thresh))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res, weak, strong


img_thresh, weak_val, strong_val = double_threshold(img_nms)

# todo Hysteresis Thresholding
def hysteresis(img, weak, strong=255):
    M, N = img.shape
    out = img.copy()

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if out[i, j] == weak:
                if ((out[i + 1, j - 1] == strong) or (out[i + 1, j] == strong) or (out[i + 1, j + 1] == strong)
                        or (out[i, j - 1] == strong) or (out[i, j + 1] == strong)
                        or (out[i - 1, j - 1] == strong) or (out[i - 1, j] == strong) or (out[i - 1, j + 1] == strong)):
                    out[i, j] = strong
                else:
                    out[i, j] = 0
    return out


img_final = hysteresis(img_thresh, weak_val, strong_val)

cv.imshow('Finalny Canny (Vlastny)', img_final)
cv.waitKey()
cv.destroyAllWindows()

cv.waitKey()