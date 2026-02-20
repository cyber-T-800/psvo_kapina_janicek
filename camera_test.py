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

def mosaic():

    # size to which we crop images
    size = 2000


    # load and crop images
    for i in range(4):
        im = cv2.imread("img/obrazok" + str(i) + ".png")
        sh = im.shape
        h = sh[0]
        hStart = int((h - size) / 2)
        w = sh[1]
        wStart = int((w - size) / 2)
        images.append(im[hStart:(hStart + size), wStart:(wStart + size), :])


    # create mosaic
    result = np.zeros([size * 2, size * 2, 3], dtype=np.uint8)

    result[0:size, 0:size, :] = images[0]
    result[0:size, size:2*size, :] = images[1]
    result[size:2*size, 0:size, :] = images[2]
    result[size:2*size, size:2*size, :] = images[3]

    cv2.imwrite("img/mosaic.png", result)

    # showing mosaic
    im = cv2.resize(result, (int(size / 5), int(size / 5)))
    cv2.imshow("mosaic", im)

    # Emboss filter, highlights edges to create a 3D effect
    # Negative values (top-left) darken pixels.
    # Positive values (bottom-right) brighten pixels.
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    result[0:size, 0:size] = cv2.filter2D(result[0:size, 0:size], -1, kernel, borderType=cv2.BORDER_REPLICATE)

    copy_2_picture = result[0:size, size:2*size].copy()
    for y in range(size):
        for x in range(size):
            rotated_row = x
            rotated_col = (size - 1) - y
            result[rotated_row, size + rotated_col] = copy_2_picture[y, x]

    result[w:2 * size, 0:size, 0] = 0
    result[size:2 * size, 0:size, 1] = 0

    print("Image info")
    print(f"Data type: {result.dtype}")
    print(f"Dimensions: {result.shape}")
    print(f"Total size {result.size}")
    cv2.imwrite("img/mosaic_after.png", result)

mosaic()

cv2.waitKey()

print('Done.')