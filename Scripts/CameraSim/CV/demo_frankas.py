import cv2
import os
import numpy as np

def detect_mark_franka(img_path, output_path):
    if not os.path.exists(img_path): return

    img = cv2.imread(img_path)
    # Pasamos a escala de grises directamente para maximizar el contraste
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. UMBRALIZACIÓN ADAPTATIVA (Clave para objetos lejanos/oscuros)
    # Buscamos píxeles con brillo superior a 130 (ajustable)
    _, mask = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    
    # 2. LIMPIEZA MORFOLÓGICA FUERTE
    # Usamos una elipse vertical para ayudar a conectar los tramos del brazo
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 25))
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # GUARDAR MÁSCARA PARA DEPURAR (Mira este archivo si falla)
    cv2.imwrite("mask_debug.png", mask_cleaned)
    
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    overlay = img.copy()
    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(h)/w

        # Filtros ultra-permisivos para la imagen lejana
        # Un Franka a esa distancia puede tener apenas 800 píxeles de área
        if area > 500 and aspect_ratio > 1.0:
            # Filtro adicional: El Franka no suele estar pegado a los bordes laterales
            if 50 < x < 1230:
                detections.append((x, y, w, h))

    # Ordenar y dibujar
    detections.sort(key=lambda d: d[0])
    for i, (x, y, w, h) in enumerate(detections):
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 3)
        cv2.putText(overlay, f"ROBOT_{i+1}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imwrite(output_path, overlay)
    print(f"##### TOTAL DETECTADO: {len(detections)} #####")

if __name__ == "__main__":
    image_path = "/home/unaiolaizolaosa/Documents/PFG/Scripts/CameraSim/CV/frankas.png" #
    output_path = "/home/unaiolaizolaosa/Documents/PFG/Scripts/CameraSim/CV/CV_results/franka_debug.png"
    detect_mark_franka(image_path, output_path)