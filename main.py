"""
=============================
Coin recognition using OpenCV
=============================

Gergő Papp-Szentannai
"""
print(__doc__)

#
# Coin recognition in OpenCV
#
# Project structure:
#   1. Detecting coins:
#       1.1 transformations,
#       1.2 morphological operations and
#       1.3 contour extraction
#   2. Recognizing coins:
#       2.1 Crop each coin
#       2.2 Use classifiers to "guess" coin (using a training data set)
#

import numpy as np
import cv2


def run_main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        roi = frame
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
        blur_ret, thresh = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#       thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)
        cv2.imshow('Gray', gray_blur)

        circles_container = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=20, maxRadius=40)

        if circles_container is None:
            continue
        else:
            circles = circles_container[0]
            #print(circles)
            # loop over the (x, y) coordinates and radius of the circles
            cimg = frame

            for (x, y, r) in circles:
                # draw the circle in the output image
                cv2.circle(cimg, (x, y), r, (0, 0, 0), 4)

            cv2.imshow('detected circles', cimg)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_main()
