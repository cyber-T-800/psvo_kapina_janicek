import numpy as np
import cv2
from camera import shutdown_camera, set_up_camera


def nothing(x):
    pass


MTX = np.array([[600.35201553, 0., 182.41220667],
                [0., 722.17797944, 192.00406333],
                [0., 0., 1.]], dtype=np.float32)

DIST = np.array([-3.98597311e-01, -4.17022141e-01, 1.09488369e-03, 3.39626032e-03, 3.40320106e+00], dtype=np.float32)

# Fyzicky odmeraná vzdialenosť kamery od podložky v cm
Z_DIST_CM = 40.0

# Priemerná ohnisková vzdialenosť v pixeloch (fx + fy) / 2
F_AVG = (MTX[0, 0] + MTX[1, 1]) / 2


def px_to_cm(pixels):
    """Prepočet pixelov na centimetre na základe kalibrácie a vzdialenosti."""
    if F_AVG == 0: return 0
    return (pixels * Z_DIST_CM) / F_AVG


track_bar_window = np.zeros((300, 512, 3), np.uint8)
cv2.namedWindow('Tracer')
cv2.createTrackbar('Param1Hough', 'Tracer', 50, 500, nothing)
cv2.createTrackbar('Param2Hough', 'Tracer', 100, 500, nothing)
cv2.createTrackbar('MinRadiusCircle', 'Tracer', 0, 500, nothing)
cv2.createTrackbar('CannyThresh1', 'Tracer', 100, 500, nothing)
cv2.createTrackbar('CannyThresh2', 'Tracer', 200, 500, nothing)
cv2.createTrackbar('Epsilon %', 'Tracer', 2, 100, nothing)
cv2.createTrackbar('Min Area', 'Tracer', 1000, 50000, nothing)


def detect_shapes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    Param1_Hough = cv2.getTrackbarPos('Param1Hough', 'Tracer')
    Param2_Hough = cv2.getTrackbarPos('Param2Hough', 'Tracer')
    Min_Radius = cv2.getTrackbarPos('MinRadiusCircle', 'Tracer')
    Canny_Thresh1 = cv2.getTrackbarPos('CannyThresh1', 'Tracer')
    Canny_Thresh2 = cv2.getTrackbarPos('CannyThresh2', 'Tracer')
    Epsilon_Pos = cv2.getTrackbarPos('Epsilon %', 'Tracer')
    Min_Area = cv2.getTrackbarPos('Min Area', 'Tracer')

    epsilon_multiplier = Epsilon_Pos / 100.0

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 150,
                               param1=Param1_Hough, param2=Param2_Hough, minRadius=Min_Radius,
                               maxRadius=0)

    if circles is not None:
        circles = np.int32(np.around(circles))
        for i in circles[0, :]:
            diameter_cm = px_to_cm(i[2] * 2)

            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 8)
            cv2.putText(frame, f"Circle: {diameter_cm:.1f}cm", (i[0] - 40, i[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5)

    edges = cv2.Canny(blurred, Canny_Thresh1, Canny_Thresh2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < Min_Area:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon_multiplier * peri, True)
        num_corners = len(approx)

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        else:
            continue

        rect = cv2.minAreaRect(contour)
        (x_c, y_c), (w_px, h_px), angle = rect
        w_cm = px_to_cm(w_px)
        h_cm = px_to_cm(h_px)

        label = ""
        if num_corners == 3:
            label = f"Tri: a={w_cm:.1f}cm"
        elif num_corners == 4:
            aspect_ratio = w_px / h_px if h_px != 0 else 0
            if 0.90 <= aspect_ratio <= 1.10:
                label = f"Square: {w_cm:.1f}cm"
            else:
                label = f"Rect: {w_cm:.1f}x{h_cm:.1f}cm"

        if label:
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 8)
            cv2.circle(frame, (cX, cY), 6, (0, 0, 255), -1)
            cv2.putText(frame, label, (cX - 50, cY), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5)

    return frame

cam, image_obj = set_up_camera()

try:
    while True:
        cam.get_image(image_obj)
        frame = image_obj.get_image_data_numpy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        processed_frame = detect_shapes(frame)

        cv2.imshow('Tracer', track_bar_window)

        img = cv2.resize(processed_frame, (800, 800))
        cv2.imshow('Detection', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        Param1_Hough_Position = cv2.getTrackbarPos('Param1Hough', 'Tracer')
        track_bar_window[:] = [Param1_Hough_Position, Param1_Hough_Position, Param1_Hough_Position]

except Exception as e:
    print(f"Chyba: {e}")
finally:
    shutdown_camera(cam)
    cv2.destroyAllWindows()