import cv2
import numpy as np

def detect(rgb_image, color):
    img_cv = cv2.cvtColor(rgb_image[..., :3], cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    
    # Rangos optimizados para evitar que el negro/sombra entre en el filtro
    ranges = {
        "red":   ([0, 160, 60], [10, 255, 255]),
        "green": ([45, 160, 60], [75, 255, 255]),
        "blue":  ([100, 160, 60], [130, 255, 255])
    }
    
    lower, upper = ranges.get(color.lower(), ([0,0,0], [180,255,255]))
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    
    # Limpieza agresiva de ruido
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    found_any = False
    for cnt in contours:
        if cv2.contourArea(cnt) > 25:
            # --- CORRECCIÓN: CENTROIDE ---
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Dibujamos un cuadro pequeño y grueso centrado en el color puro
                offset = 12
                cv2.rectangle(img_cv, (cX-offset, cY-offset), (cX+offset, cY+offset), (0, 255, 255), 4)
                found_any = True
                
    return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), found_any