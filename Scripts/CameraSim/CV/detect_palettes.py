# script to detect pallets using CV, we will look for the object centroid, this script will be called from the camera detection main script


import cv2
import numpy as np

def detect_palette(rgb_image, color):
    img_cv = cv2.cvtColor(rgb_image[..., :3], cv2.COLOR_RGB2BGR)
    h, w, _ = img_cv.shape
    overlay = img_cv.copy()

    # mask to remove the franka robot part of the image the camera_02 captures
    exclusion_mask = np.ones((h, w), dtype=np.uint8) * 255
    cv2.rectangle(exclusion_mask, (440, 440), (840, h), 0, -1)
    
    ranges = {
        "red":   ([0, 70, 20],     [15, 255, 255]),
        "blue":  ([95, 70, 20],    [130, 255, 255]),
        "black": ([0, 0, 0],       [180, 255, 75]) 
    }
    
    lower, upper = ranges.get(color.lower(), ([0,0,0], [180,255,255]))
    
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    
    mask = cv2.bitwise_and(mask, exclusion_mask)
    
    kernel = np.ones((15, 15), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    found_any = False
    
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        aspect_ratio = float(w_box) / h_box
        
        if area > 1500 and (0.5 < aspect_ratio < 2.0):
            valid_contours.append(cnt)

    if valid_contours:
        largest_cnt = max(valid_contours, key=cv2.contourArea)
        M = cv2.moments(largest_cnt)
        
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # rectangle marking + a circle in the middle because of the holes the pallets have
            x, y, wb, hb = cv2.boundingRect(largest_cnt)
            cv2.rectangle(overlay, (x, y), (x + wb, y + hb), (0, 255, 255), 4)
            cv2.circle(overlay, (cX, cY), 10, (0, 255, 255), -1)
            
            found_any = True

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), found_any