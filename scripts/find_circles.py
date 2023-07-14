import cv2
import numpy as np

def find_circles(img, min_area=1, max_area=250, aspect_ratio_threshold=2.5, circularity_threshold=0.2,
                 lr1=[0, 40, 20], ur1=[40, 255, 255], lr2=[160, 40, 20], ur2=[210, 255, 255], 
                 lb=[90, 100, 40], ub=[150, 255, 255], lg=[55, 29, 0], ug=[90, 255, 171]):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define color masks for red, blue, and green
    lower_red = np.array(lr1)
    upper_red = np.array(ur1)
    mask_red_1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array(lr2)
    upper_red = np.array(ur2)
    mask_red_2 = cv2.inRange(hsv, lower_red, upper_red)

    mask_red = mask_red_1 | mask_red_2

    lower_blue = np.array(lb)
    upper_blue = np.array(ub)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_green = np.array(lg)
    upper_green = np.array(ug)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    masks = {"Red": mask_red, "Blue": mask_blue, "Green": mask_green}
    colors = {"Red": (0, 0, 255), "Blue": (255, 0, 0), "Green": (0, 255, 0)}

    img_all_valid_contours = img.copy()
    
    # Store all contours with colours extracted from the loaded image
    contours_with_color = []

    # Find contours for each color mask
    for color, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                valid_contours.append(cnt)

        # Filter contours based on aspect ratio
        square_contours = []
        for cnt in valid_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if 1 / aspect_ratio_threshold < aspect_ratio < aspect_ratio_threshold:
                square_contours.append(cnt)

        # Filter contours based on circularity
        round_contours = []
        for cnt in square_contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > circularity_threshold:
                round_contours.append(cnt)

        # Draw valid round contours on the image
        cv2.drawContours(img_all_valid_contours, round_contours, -1, colors[color], 1)
    
        # Store contour along with its color and centroid in contours_with_color and return it
        for cnt in round_contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                contours_with_color.append((cnt, color, cX, cY))

    return img_all_valid_contours, contours_with_color