import numpy as np
import cv2
import imageio.v2 as iio

def nothing(x):
    pass

frame = iio.imread("zadanie2/test.jpg")

if frame is None:
    print(f"Error: Could not load image from {image_path}. Check the file path.")
    exit()

original_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(original_frame, cv2.COLOR_RGB2GRAY)
blurred = cv2.medianBlur(gray, 5)

tracebar_window = np.zeros((300,512,3), np.uint8)
cv2.namedWindow('Tracebar')
cv2.createTrackbar('Param1Hough', 'Tracebar', 0, 1000, nothing)

circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 150,
                            param1=50, param2=60, minRadius=45, maxRadius=0)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 3)
        cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
        cv2.putText(frame, "Circle", (i[0] - 20, i[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

edges = cv2.Canny(blurred, 10,  10)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    epsilon = 0.1 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    num_corners = len(approx)

    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    x, y, w, h = cv2.boundingRect(approx)

    label = ""
    if num_corners == 3:
        label = "Triangle"
    elif num_corners == 4:
        aspect_ratio = float(w) / h
        label = "Square" if 0.98 <= aspect_ratio <= 1.02 else "Rectangle"

    if label:
        cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)
        cv2.circle(frame, (cX, cY), 2, (0, 0, 255), 3)
        cv2.putText(frame, label, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    while True:
        cv2.imshow('Tracebar', tracebar_window)
        cv2.imshow('Detection', frame)

        k = cv2.waitKey(500)
        if k == 27:
            break
        elif k == -1:
            continue

        Param1_Hough_Position = cv2.getTrackbarPos('Param1_Hough','Tracebar')

        tracebar_window[:] = [Param1_Hough_Position]

cv2.destroyAllWindows()