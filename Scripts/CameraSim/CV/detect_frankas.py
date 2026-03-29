# script to detect frankas using CV, we will look for the object centroid, this script will be called from the camera detection main script
import cv2 
import numpy as np

def detect(rgb_image, color):
    img_cv = cv2.cvtColor(rgb_image[..., :3], cv2.COLOR_RGB2BGR) # transform rgb -> bgr
    h, w, _ = img_cv.shape
    overlay = img_cv.copy()

    hls = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HLS)
    lower_white = np.array([0, 200, 0])
    upper_white = np.array([180, 255, 60])
    mask_white = cv2.inRange(hls, lower_white, upper_white) # filter the white between ranges

    # create mask to ignore other objects
    mask = np.zeros((h,w), dtype=np.uint8)
    cv2.rectangle(mask, (0,0), (int(w*.35), h), 255, -1)
    cv2.rectangle(mask, (int(w*.65), 0), (w,h), 255, -1)
    final_mask = cv2.bitwise_and(mask_white, mask)

    kernel = np.ones((25, 25), np.uint8)
    mask_cleaned = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found_any = False 
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 3500:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                x, y, wb, hb = cv2.boundingRect(contour)
                cv2.rectangle(overlay, (x,y), (x+wb, y+hb), (0,255, 0), 4)
                cv2.circle(overlay, (cX, cY), 12, (0,255, 0), -1)
                found_any = True

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), found_any # and we return back the image from bgr -> rgb