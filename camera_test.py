from ximea import xiapi
import cv2
import numpy as np
### runn this command first echo 0|sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb  ###

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
    cam.set_exposure(50000)
    cam.set_param("imgdataformat","XI_RGB32")
    cam.set_param("auto_wb",1)

    print('Exposure was set to %i us' %cam.get_exposure())

    # create instance of Image to store image data and metadata
    img = xiapi.Image()

    # start data acquisitionq
    print('Starting data acquisition...')
    cam.start_acquisition()

    for i in range(4):
        #get data and pass them from camera to img
        cam.get_image(img)
        image = img.get_image_data_numpy()

        cv2.imwrite("img/obrazok" + str(i) + ".png", image)
        image = cv2.resize(image, (240, 240))
        cv2.imshow("test", image)
        cv2.waitKey()

    # stop data acquisition
    print('Stopping acquisition...')
    cam.stop_acquisition()

    # stop communication
    cam.close_device()
    print('Camera stopped.')

#photoshoot()

# load images

images = []

# loading images
for i in range(4):
    images.append(cv2.imread("img/obrazok" + str(i) + ".png"))

def mosaic():
    sh = images[0].shape
    w = sh[0]
    h = sh[1]
    result = np.zeros([w * 2, h * 2, 3], dtype=np.uint8)

    result[0:w, 0:h, :] = images[0]
    result[0:w, h:2*h, :] = images[1]
    result[w:2*w, 0:h, :] = images[2]
    result[w:2*w, h:2*h, :] = images[3]

    cv2.imwrite("img/mosaic.png", result)

mosaic()
# print data
# data_raw = img.get_image_data_raw()
#
# #transform data to list
# data = list(data_raw)
# print('Image number: ' + str(i))
# print('Image width (pixels):  ' + str(img.width))
# print('Image height (pixels): ' + str(img.height))
# print('First 10 pixels: ' + str(data[:10]))
# print('\n')


print('Done.')