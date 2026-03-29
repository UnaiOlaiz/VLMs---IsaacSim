import cv2
import os
import numpy as np

def detect_only_frankas(img_path, output_path):
    """
    Script especializado para detectar exclusivamente robots Franka blancos.
    Ignora palets de colores, Jetbots grises y rejillas negras.
    """
    if not os.path.exists(img_path): return

    img = cv2.imread(img_path)
    h, w, _ = img.shape
    overlay = img.copy()

    # --- ESTRATEGIA 1: FILTRADO POR LUMINANCIA (BLANCO ROBOT) ---
    # Convertimos a HLS (Hue, Lightness, Saturation) para aislar el blanco puro
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    
    # El blanco del robot tiene una Lightness (L) muy alta (>200) 
    # y una Saturation (S) muy baja (<50) porque no tiene color.
    lower_white = np.array([0, 200, 0])
    upper_white = np.array([180, 255, 60])
    mask_white = cv2.inRange(hls, lower_white, upper_white)

    # --- ESTRATEGIA 2: MÁSCARA ESPACIAL (ZONAS DE FRANKAS) ---
    # Solo buscamos en los extremos laterales donde sabemos que están las bases
    exclusion_mask = np.zeros((h, w), dtype=np.uint8)
    # Zona Franka Izquierda
    cv2.rectangle(exclusion_mask, (0, 0), (int(w*0.3), h), 255, -1)
    # Zona Franka Derecha
    cv2.rectangle(exclusion_mask, (int(w*0.7), 0), (w, h), 255, -1)
    
    # Combinamos el filtro de color blanco con la posición espacial
    final_mask = cv2.bitwise_and(mask_white, exclusion_mask)

    # --- ESTRATEGIA 3: LIMPIEZA MORFOLÓGICA ---
    # Usamos un kernel grande para unir las partes del brazo (hombro, codo, muñeca)
    kernel = np.ones((30, 30), np.uint8)
    mask_cleaned = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Un Franka a esta distancia debe tener un área considerable
        if area > 4000: 
            x, y, wb, hb = cv2.boundingRect(contour)
            
            # Calculamos el centroide de la masa blanca
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                
                # Dibujamos marcador específico para el Robot
                cv2.rectangle(overlay, (x, y), (x + wb, y + hb), (0, 255, 0), 3) # Verde para Robots
                cv2.circle(overlay, (cX, cY), 15, (0, 255, 0), -1)
                #cv2.putText(overlay, "FRANKA ROBOT", (x, y-15), 
                            # cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                count += 1

    cv2.imwrite(output_path, overlay)
    print(f"##### DETECCIÓN DE ROBOTS: {count} FRANKAS LOCALIZADOS #####")



if __name__ == "__main__":
    image_path = "/home/unaiolaizolaosa/Documents/PFG/Scripts/CameraSim/CV/frankas.png"
    output_path = "/home/unaiolaizolaosa/Documents/PFG/Scripts/CameraSim/CV/CV_results/frankas_clean.png"
    detect_only_frankas(image_path, output_path)