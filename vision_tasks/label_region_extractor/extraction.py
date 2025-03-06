import cv2
import numpy as np

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    """
    1. Convert the image into a depth map of green values (only green areas will be non-zero).
    2. Extract contours of all zero-value regions in the depth map.
    3. Filter contours based on the color inside (keep only those with predominantly white interiors).
    4. Further filter based on the colors to the left and right of the contour (keep only those that have green on both sides).
    """

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the green color range (adjust if needed)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    # Create a mask for green areas
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Create a depth map where only green areas are nonzero
    depth_map = np.zeros_like(mask_green)
    depth_map[mask_green > 0] = 255  # Set green areas to 255 (non-zero)

    # cv2.imshow('Green Depth Map', depth_map)
    
    # Invert the depth map to find zero-value regions (non-green areas)
    mask_not_green = cv2.bitwise_not(depth_map)

    # Find contours of the zero-value regions
    contours, _ = cv2.findContours(mask_not_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []

    for contour in contours:
        # Create a mask for the current contour
        mask_contour = np.zeros_like(mask_not_green)
        cv2.drawContours(mask_contour, [contour], -1, 255, thickness=cv2.FILLED)

        # Extract the region inside the contour from the original frame
        region = cv2.bitwise_and(frame, frame, mask=mask_contour)

        # Convert to grayscale and compute the mean intensity
        gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        mean_intensity = cv2.mean(gray_region, mask=mask_contour)[0]

        # Consider a contour valid if the mean intensity suggests a white region
        if mean_intensity > 100:  # White has high intensity (near 255)
            valid_contours.append(contour)

    final_contours = []

    for contour in valid_contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Define left and right regions for checking green color
        margin = max(5, w // 10)  # Set a small margin based on contour size
        left_region = frame[y:y+h, max(0, x-margin):x]
        right_region = frame[y:y+h, x+w:min(frame.shape[1], x+w+margin)]

        # Convert both regions to HSV
        hsv_left = cv2.cvtColor(left_region, cv2.COLOR_BGR2HSV) if left_region.size > 0 else None
        hsv_right = cv2.cvtColor(right_region, cv2.COLOR_BGR2HSV) if right_region.size > 0 else None

        # Check if both left and right regions contain green
        left_mask = cv2.inRange(hsv_left, lower_green, upper_green) if hsv_left is not None else np.array([])
        right_mask = cv2.inRange(hsv_right, lower_green, upper_green) if hsv_right is not None else np.array([])

        if left_mask.any() and right_mask.any():
            final_contours.append(contour)

    # Draw the final filtered contours
    cv2.drawContours(frame, final_contours, -1, (0, 0, 255), 2)

    # Display the result
    cv2.imshow('Filtered Contours', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
