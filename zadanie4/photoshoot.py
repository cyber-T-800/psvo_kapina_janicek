from time import sleep

from ximea import xiapi
import cv2
import numpy as np

def photoshoot():
    # create instance for first connected camera
    cam = xiapi.Camera()

    # start communication
    # to open specific device, use:
    # cam.open_device_by_SN('41305651')
    # (open by serial number)
    print('Opening first camera...')
    cam.open_device()

    # settings
    cam.set_exposure(10000)
    cam.set_param("imgdataformat","XI_RGB32")
    cam.set_param("auto_wb",1)

    print('Exposure was set to %i us' %cam.get_exposure())

    # create instance of Image to store image data and metadata
    img = xiapi.Image()

    # start data acquisitionq
    print('Starting data acquisition...')
    cam.start_acquisition()

    img_shoted = 1
    timing = 0

    while img_shoted < 300:
        #get data and pass them from camera to img
        cam.get_image(img)
        image = img.get_image_data_numpy()

        im = cv2.resize(image, None, fx=0.45,fy=0.45,interpolation=cv2.INTER_LINEAR)
        cv2.imshow("test", im)

        if (cv2.waitKey(12) == 32):
            break

        if timing % 5 == 0:
            cv2.imwrite("img/obrazok" + str(img_shoted) + ".png", im)
            img_shoted += 1
            print(img_shoted)
        timing+= 1

    # stop data acquisition
    print('Stopping acquisition...')
    cam.stop_acquisition()

    # stop communication
    cam.close_device()
    print('Camera stopped.')

photoshoot()

