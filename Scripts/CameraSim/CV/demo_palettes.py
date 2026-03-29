import cv2
import os
import numpy as np

def detect_mark_palettes(img_path, output_path):
    if not os.path.exists(img_path): return

    img = cv2.imread(img_path)
    h, w, _ = img.shape
    
    # --- PASO CRÍTICO: MÁSCARA DE EXCLUSIÓN PARA EL ROBOT ---
    # Creamos un rectángulo negro sobre el robot (ajusta estos valores si es necesario)
    # Basado en tu imagen: x desde 450 a 830, y desde 450 hasta el final
    exclusion_zone = img.copy()
    cv2.rectangle(exclusion_zone, (440, 440), (840, h), (0, 0, 0), -1)
    
    # Usamos la imagen con el robot "borrado" para la detección
    hsv = cv2.cvtColor(exclusion_zone, cv2.COLOR_BGR2HSV)
    overlay = img.copy()

    color_ranges = {
        "red":   ([0, 70, 20],     [15, 255, 255]),
        "blue":  ([95, 70, 20],    [130, 255, 255]),
        "black": ([0, 0, 0],       [180, 255, 70]) # Ajustado para paletas oscuras
    }

    count = 0
    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        # Clausura morfológica para unir la rejilla de la paleta
        kernel = np.ones((15, 15), np.uint8)
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w_box, h_box = cv2.boundingRect(contour)
            
            # Filtro de área: Las paletas son grandes
            if area < 1500: continue 
            
            # Filtro de forma: Relación de aspecto (deben ser rectangulares/cuadradas)
            aspect_ratio = float(w_box)/h_box
            if not (0.5 < aspect_ratio < 2.0): continue

            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                
                # Dibujamos en la imagen original (sin el parche negro)
                cv2.rectangle(overlay, (x, y), (x + w_box, y + h_box), (0, 255, 255), 3)
                cv2.circle(overlay, (cX, cY), 10, (0, 255, 255), -1)
                """
                cv2.putText(overlay, f"PALETTE {color_name.upper()}", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                """
                count += 1

    cv2.imwrite(output_path, overlay)
    print(f"##### EXCLUSIÓN ACTIVA: {count} PALETAS REALES #####")

if __name__ == "__main__":
    image_path = "/home/unaiolaizolaosa/Documents/PFG/Scripts/CameraSim/CV/palettes.png"
    output_path = "/home/unaiolaizolaosa/Documents/PFG/Scripts/CameraSim/CV/CV_results/palette_clean.png"
    detect_mark_palettes(image_path, output_path)