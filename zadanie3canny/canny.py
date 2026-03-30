import time

import numpy as np
import cv2 as cv
import convolution as conv

image = cv.imread('messi.png', cv.IMREAD_GRAYSCALE)


start = time.time()
can = cv.Canny(image, 100, 200)
end = time.time()
cv.imwrite('prezentacia/img_c_3.png', can)

print("canny elapsed time: ", end - start)
cv.imshow('canny', can)


start = time.time()
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

imgHranyX = conv.convolve(img.astype(np.float32), Gx)
imgHranyY = conv.convolve(img.astype(np.float32), Gy)

edgeGradient = np.hypot(imgHranyX, imgHranyY)
edgeGradient[np.isnan(edgeGradient)] = 0

edgeGradient = edgeGradient / np.max(edgeGradient) * 255
edgeGradient = edgeGradient.astype(np.uint8)

imgHranyX[imgHranyX == 0] = 0.000001
gAngle = np.atan2(imgHranyY, imgHranyX)
gAngle[np.isnan(gAngle)] = 0

gAngle = gAngle * 180 / np.pi
gAngle[gAngle < 0] = 180 + gAngle[gAngle < 0]


def non_max_suppression(magnitude, angle_matrix):
    M, N = magnitude.shape
    Z = np.zeros((M, N), dtype=np.float32)

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
    return Z.astype(np.uint8)


img_nms = non_max_suppression(edgeGradient, gAngle)

def double_threshold(img, low_ratio=0.25, high_ratio=0.18):
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


end = time.time()
print("vlastny canny elapsed time: ", end - start)
cv.imwrite('prezentacia/img_f_3.png', img_final)



cv.imshow('Finalny Canny (Vlastny)', img_final)
cv.waitKey()
cv.destroyAllWindows()

cv.waitKey()