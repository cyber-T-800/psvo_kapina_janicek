import glob

import cv2 as cv
import numpy as np
import camera

sizeX, sizeY = 7, 5

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((sizeX * sizeY, 3), np.float32)
objp[:, :2] = np.mgrid[0:sizeX, 0:sizeY].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('*.jpg')


def photoshoot():
    cam, image = camera.set_up_camera()

    counter, img_counter = 0, 0
    last = -10

    while counter < 20:
        cam.get_image(image)
        img = image.get_image_data_numpy()
        img = cv.resize(img, (400, 400))

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (sizeX, sizeY), None)

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

            cv.imshow('img', img)
            cv.waitKey(100)
        else:
            print('No chessboard corners found!')
            cv.imshow('img-no', img)
            cv.waitKey(100)

            counter += 1

    camera.shutdown_camera(cam)
    cv.destroyAllWindows()

def calibrate():
    gray = []
    for i in range(21):

        img = cv.imread("./img/obrazok" + str(i) + ".png")

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (sizeX, sizeY), None)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners2)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs


def draw(mtx, dist, rvecs, tvecs):
    for i in range(21):
        img = cv.imread('./img/obrazok' + str(i) + '.png')
        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # undistort
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image
        x, y, w, h = roi

        gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (sizeX, sizeY), None)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        cv.drawChessboardCorners(dst, (sizeX, sizeY), corners2, ret)

        dst = dst[y:y + h, x:x + w]
        print('dst: ', dst.shape)
        cv.imshow('img', img)
        cv.imshow('dst', dst)

        mean_error = 0
        for j in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(objpoints[j], rvecs[j], tvecs[j], mtx, dist)
            error = cv.norm(imgpoints[j], imgpoints2, cv.NORM_L2) / len(imgpoints2)
            mean_error += error

        print("total error: {}".format(mean_error / len(objpoints)))

        cv.waitKey(0)


ret, mtx, dist, rvecs, tvecs = calibrate()

print("matica parametrov:")

print(mtx)

print('parametre kamery:')
print('fx:', mtx[0, 0])
print('fy:', mtx[1, 1])
print('cx:', mtx[0, 2])
print('cy:', mtx[1, 2])

print('disorčné paramtre: ')
print(dist)