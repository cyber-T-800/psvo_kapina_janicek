import cv2 as cv
from camera import set_up_camera, shutdown_camera

def detect_shapes(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (9, 9), 1.5)

    circles = cv.HoughCircles(
        blurred,
        cv.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=120,
        param2=40,
        minRadius=20,
        maxRadius=300
    )

    if circles is not None:
        circles = circles[0].astype(int)
        for x, y, r in circles:
            cv.circle(frame, (x, y), r, (255, 0, 0), 3)
            cv.circle(frame, (x, y), 3, (0, 0, 255), -1)
            cv.putText(frame, "Kruh", (x - 20, y - 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    edged = cv.Canny(blurred, 100, 200)
    contours, _ = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv.contourArea(cnt)

        if area < 2000:
            continue

        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.02 * peri, True)

        M = cv.moments(cnt)
        if M["m00"] == 0:
            continue

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        shape_name = "Neznamy"
        num_vertices = len(approx)

        if num_vertices == 3:
            shape_name = "Trojuholnik"
        elif num_vertices == 4:
            x, y, w, h = cv.boundingRect(approx)
            aspect_ratio = float(w) / h
            shape_name = "Stvorec" if 0.9 <= aspect_ratio <= 1.1 else "Obdlznik"

        else:
            continue

        cv.drawContours(frame, [approx], -1, (0, 255, 0), 2)
        cv.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
        cv.putText(frame, shape_name, (cX - 20, cY - 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


cam, img = set_up_camera()

try:
    while True:
        cam.get_image(img)
        frame = img.get_image_data_numpy()
        frame = cv.cvtColor(frame, cv.COLOR_RGBA2BGR)
        processed_frame = detect_shapes(frame)
        cv.imshow('XIMEA Úloha 2 - Detekcia', processed_frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Chyba: {e}")

finally:
    shutdown_camera(cam)
    cv.destroyAllWindows()