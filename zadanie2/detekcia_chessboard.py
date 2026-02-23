import glob

import cv2 as cv
import numpy as np
import camera

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('*.jpg')

cam, image = camera.set_up_camera()

counter, img_counter = 0, 0
last = -10

while True:
    cam.get_image(image)
    img = image.get_image_data_numpy()
    img = cv.resize(img, (400, 400))

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7, 5), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        print('Found chessboard corners...')
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        if counter > last + 10:
            cv.imwrite("./img/obrazok" + str(img_counter) + ".png", img)
            last = counter
            img_counter += 1
            print("ulozene")

        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,5), corners2, ret)

        cv.imshow('img', img)
        cv.waitKey(100)

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    else:
        print('No chessboard corners found!')
        cv.imshow('img-no', img)
        cv.waitKey(100)

    counter += 1

camera.shutdown_camera(cam)
cv.destroyAllWindows()
