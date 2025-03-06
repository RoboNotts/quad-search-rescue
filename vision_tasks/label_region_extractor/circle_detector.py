import cv2
import numpy as np

def is_circle_inside_contour(circle, contour, mask):
    """
    Check if a detected circle is fully inside the given contour.
    
    Parameters:
        circle (tuple): (x, y, radius) of the detected circle.
        contour (numpy.ndarray): The contour to check against.
        mask (numpy.ndarray): Binary mask of the contour region.

    Returns:
        bool: True if the circle is fully inside the contour, False otherwise.
    """
    x, y, r = circle
    # center of the circle should be in the contour
    if mask[y, x] == 0:
        return False

    # check whether the boundary of the circle in inside the contour
    for angle in range(0, 360, 10): 
        theta = np.deg2rad(angle)
        px = int(x + r * np.cos(theta))
        py = int(y + r * np.sin(theta))

        if mask[py, px] == 0:
            return False

    return True

def detect_largest_circle(image, contour):
    """
    Detects the largest circle inside a given contour using Hough Circle Transform.
    
    Parameters:
        image (numpy.ndarray): The input image (BGR format).
        contour (numpy.ndarray): The contour within which circles are detected.
    
    Returns:
        tuple: (x, y, r) of the largest detected circle, or None if no circle is found.
    """
    # make a mask for the region inside the contour
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # extract ROI (regions of interest)
    roi = cv2.bitwise_and(image, image, mask=mask)
    
    # convert to grey image
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    gray_roi = cv2.GaussianBlur(gray_roi, (9, 9), 2)

    # detect circles
    circles = cv2.HoughCircles(
        gray_roi, 
        cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
        param1=50, param2=30, minRadius=10, maxRadius=200
    )

    valid_circles = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0]:
            if is_circle_inside_contour(circle, contour, mask):
                valid_circles.append(tuple(circle))  # filter the circles in the contour

        if valid_circles:
            largest_circle = max(valid_circles, key=lambda c: c[2])  # sort by radius, find the biggest one
            return tuple(largest_circle)  # (x, y, radius)

    return None


