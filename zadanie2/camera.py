from ximea import xiapi


def set_up_camera():
    cam = xiapi.Camera()
    print('Opening first camera...')
    cam.open_device()

    # settings
    cam.set_exposure(50000)
    cam.set_param("imgdataformat", "XI_RGB32")
    cam.set_param("auto_wb", 1)

    print('Exposure was set to %i us' % cam.get_exposure())

    # create instance of Image to store image data and metadata
    image = xiapi.Image()

    # start data acquisitionq
    print('Starting data acquisition...')
    cam.start_acquisition()


    return cam, image

def shutdown_camera(cam):
    # stop data acquisition
    print('Stopping acquisition...')
    cam.stop_acquisition()

    # stop communication
    cam.close_device()
    print('Camera stopped.')
