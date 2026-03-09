import numpy as np
import cv2
from camera import shutdown_camera, set_up_camera

MTX = np.array([[600.35201553,   0.,         182.41220667],
                   [  0.,         722.17797944, 192.00406333],
                   [  0.,           0.,           1.        ]], dtype=np.float32)

DIST = np.array([-3.98597311e-01, -4.17022141e-01,  1.09488369e-03,  3.39626032e-03, 3.40320106e+00], dtype=np.float32)  # 'dist' hodnoty todo

# Fyzicky odmeraná vzdialenosť kamery od podložky v cm
Z_DIST_CM = 40.0

# Priemerná ohnisková vzdialenosť v pixeloch (fx + fy) / 2
F_AVG = (MTX[0, 0] + MTX[1, 1]) / 2


def px_to_cm(pixels):
    """Prepočet pixelov na centimetre na základe kalibrácie a vzdialenosti."""
    if F_AVG == 0: return 0
    return (pixels * Z_DIST_CM) / F_AVG


def detect_shapes(frame):
    # 1. Odstránenie skreslenia šošovky (undistort)
    frame = cv2.undistort(frame, MTX, DIST, None, MTX)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

    edges = cv2.Canny(blurred, 30, 90)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 150,
                               param1=50, param2=60, minRadius=45, maxRadius=0)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Výpočet priemeru v cm
            diameter_cm = px_to_cm(i[2] * 2)

            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 3)
            cv2.putText(frame, f"Circle: {diameter_cm:.1f}cm", (i[0] - 40, i[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000: continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        num_corners = len(approx)

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        else:
            continue

        # Výpočet rozmerov cez minAreaRect (otočený obdĺžnik je presnejší pre meranie)
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
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)
            cv2.circle(frame, (cX, cY), 2, (0, 0, 255), 3)
            cv2.putText(frame, label, (cX - 50, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

cam, image_obj = set_up_camera()

try:
    while True:
        cam.get_image(image_obj)
        frame = image_obj.get_image_data_numpy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        processed_frame = detect_shapes(frame)
        cv2.imshow('XIMEA Úloha 2 - Detekcia a Meranie', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"Chyba: {e}")
finally:
    shutdown_camera(cam)
    cv2.destroyAllWindows()