import numpy as np
import cv2

def nothing(x):
    pass

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

track_bar_window = np.zeros((300,512,3), np.uint8)
cv2.namedWindow('Tracer')
cv2.createTrackbar('Param1Hough', 'Tracer', 50, 500, nothing)
cv2.createTrackbar('Param2Hough', 'Tracer', 100, 500, nothing)
cv2.createTrackbar('MinRadiusCircle', 'Tracer', 0, 500, nothing)
cv2.createTrackbar('CannyThresh1', 'Tracer', 10, 500, nothing)
cv2.createTrackbar('CannyThresh2', 'Tracer', 10, 500, nothing)
cv2.createTrackbar('Epsilon %', 'Tracer', 10, 100, nothing)

while True:
    ret, frame = cap.read()
    original_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(original_frame, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    # INCREASING IT: Makes edge detection stricter.
    # It will ignore faint outlines and only look for very harsh, high-contrast edges.
    Param1_Hough_Position = cv2.getTrackbarPos('Param1Hough', 'Tracer')
    # INCREASING IT: Demands a more mathematically perfect circle.
    # It eliminates false positives (hallucinated circles), but if set too high,
    # it misses real circles that are slightly warped or blurry.
    Param2_Hough_Position = cv2.getTrackbarPos('Param2Hough', 'Tracer')
    Min_Radius_Circle = cv2.getTrackbarPos('MinRadiusCircle', 'Tracer')
    # INCREASING THEM: Makes shape edge detection stricter.
    # Thresh2 is the main strictness (requires sharper contrast to start an edge).
    # Thresh1 is the linking strictness (drops weaker, fainter connecting lines).
    # If you increase these too much,
    # your squares and triangles will break apart into un-connectable dots.
    Canny_Thresh1_Pos = cv2.getTrackbarPos('CannyThresh1', 'Tracer')
    Canny_Thresh2_Pos = cv2.getTrackbarPos('CannyThresh2', 'Tracer')
    # INCREASING IT: Makes the shape approximation more aggressive and loose.
    # It smooths out jagged, bumpy edges and snaps them into fewer corners.
    # If increased too much, a jagged square might accidentally get simplified into a triangle!
    Epsilon_Pos = cv2.getTrackbarPos('Epsilon %', 'Tracer')

    epsilon_multiplier = Epsilon_Pos / 100.0

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 150,
                                param1=Param1_Hough_Position, param2=Param2_Hough_Position, minRadius=Min_Radius_Circle, maxRadius=0)

    if circles is not None:
        circles = np.uint32(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 3)
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
            cv2.putText(frame, "Circle", (i[0] - 20, i[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    edges = cv2.Canny(blurred, Canny_Thresh1_Pos,  Canny_Thresh2_Pos)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000:
            continue

        epsilon = epsilon_multiplier * cv2.arcLength(contour, True)
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
            label = "Square" if 0.85 <= aspect_ratio <= 1.15 else "Rectangle"

        if label:
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)
            cv2.circle(frame, (cX, cY), 2, (0, 0, 255), 3)
            cv2.putText(frame, label, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


    cv2.imshow('Tracer', track_bar_window)
    # cv2.imshow('Detection', frame)
    img = cv2.resize(frame, (700, 700))
    cv2.imshow('Detection', img)

    k = cv2.waitKey(500)
    if k == 27:
        break
    elif k == -1:
        continue

    track_bar_window[:] = [Param1_Hough_Position, Param1_Hough_Position, Param1_Hough_Position]

cv2.destroyAllWindows()