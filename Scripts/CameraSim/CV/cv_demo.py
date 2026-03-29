# Demo script updated: Detection of the cubes by CV -> send results to VLM
import cv2
import os
import numpy as np

def detect_mark_cubes(img_path, output_path):
    if not os.path.exists(img_path):
        print(f"Error: No image at {img_path}")
        return

    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    overlay = img.copy()

    color_ranges = {
        "red":   ([0, 150, 50], [10, 255, 255]),
        "green": ([40, 150, 50], [80, 255, 255]),
        "blue":  ([100, 150, 50], [140, 255, 255])
    }

    count = 0

    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper)) # create color mask with defined color ranges
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 40: 
                continue 

            x, y, w, h = cv2.boundingRect(contour)
            
            aspect_ratio = float(w)/h
            if 0.5 < aspect_ratio < 1.8:
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)
                #cv2.putText(overlay, color_name, (x, y-5), 
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # print(f"{color_name.upper()} detected (without shadow): x={x}, y={y}, Area={int(area)}")
                count += 1

    # Save result
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    out_file = output_path if output_path.endswith('.png') else os.path.join(output_path, "result_no_shadow.png")
    
    cv2.imwrite(out_file, overlay)
    print(f"##### DETECTION FINISHED, OBJECTS FOUND: {count} #####")

if __name__ == "__main__":
    image_path = "/home/unaiolaizolaosa/Documents/PFG/Scripts/Control/camera_data/debug_full.png"
    output_path = "/home/unaiolaizolaosa/Documents/PFG/Scripts/CameraSim/CV/CV_results/result_clean.png"
    detect_mark_cubes(image_path, output_path)