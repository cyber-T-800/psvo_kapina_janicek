import cv2
import cv2 as cv
import numpy as np
import camera

#start
cam, image = camera.set_up_camera()

while True:
    cam.get_image(image)
    img = image.get_image_data_numpy()
    img = cv.resize(img, (400, 400))

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # lower_blue = np.array([110,50,50])
    # upper_blue = np.array([130,255,255])
    #
    # lower_white = np.array([0,0,235])
    # upper_white = np.array([255,255,255])

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([20, 255, 255])

    #mask = cv.inRange(hsv, lower_blue, upper_blue)
    mask = cv.inRange(hsv, lower_red, upper_red )

    res = cv.bitwise_and(img, img)
    res[mask>0]=(255,0,0, 255)

    cv.imshow('mask',mask)
    cv.imshow('image', img)
    cv.imshow('res',res)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break


# end
camera.shutdown_camera(cam)