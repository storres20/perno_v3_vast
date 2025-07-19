from ultralytics import YOLO
import cv2
import numpy as np
import math
import json
import os

model = YOLO("runs/segment/perno_seg_v1/weights/best.pt")
cap = cv2.VideoCapture(0)

# Archivo de calibración
calib_file = "calibracion.json"
mm_per_pixel = None
min_ancho_vastago = None
mensaje_calibracion = ""

# Cargar calibración si existe
if os.path.exists(calib_file):
    with open(calib_file, "r") as f:
        mm_per_pixel = json.load(f)["mm_per_pixel"]
    mensaje_calibracion = f"✅ Calibrado: {mm_per_pixel:.4f} mm/pixel"
else:
    mensaje_calibracion = "🔧 Coloca un perno M3 (3mm) para calibrar..."

# Clasificación por ancho
def clasificar_perno(diam_mm):
    if 2.6 <= diam_mm <= 3.4:
        return "M3"
    elif 3.5 <= diam_mm <= 4.4:
        return "M4"
    elif 4.5 <= diam_mm <= 5.4:
        return "M5"
    elif 5.5 <= diam_mm <= 6.4:
        return "M6"
    elif 7.5 <= diam_mm <= 8.5:
        return "M8"
    else:
        return "Desconocido"

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.25, task="segment", show=False)[0]
    masks = results.masks.data if results.masks else []

    if len(masks) == 0:
        min_ancho_vastago = None

    for mask in masks:
        mask_np = mask.cpu().numpy().astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            corte_y = int(y + 0.4 * h)
            vastago_mask = mask_np[corte_y:y + h, x:x + w]

            sub_contours, _ = cv2.findContours(vastago_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for scnt in sub_contours:
                sx, sy, sw, sh = cv2.boundingRect(scnt)
                sx += x
                sy += corte_y

                # Ajustamos el rango para permitir vástagos grandes en px (cerca)
                if mm_per_pixel is None and 10 <= sw <= 300:
                    mm_per_pixel = 3.0 / sw
                    with open(calib_file, "w") as f:
                        json.dump({"mm_per_pixel": mm_per_pixel}, f)
                    mensaje_calibracion = f"✅ Calibrado: {mm_per_pixel:.4f} mm/pixel"
                    print(mensaje_calibracion)

                if mm_per_pixel:
                    ancho_mm = sw * mm_per_pixel
                    clasificacion = clasificar_perno(ancho_mm)

                    if ancho_mm > 2.5:
                        if min_ancho_vastago is None or ancho_mm < min_ancho_vastago:
                            min_ancho_vastago = ancho_mm

                    # Mostrar resultado
                    cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (255, 255, 0), 2)
                    cv2.putText(frame, f"{clasificacion} ({ancho_mm:.2f} mm)", (sx, sy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    # También puedes ver la máscara segmentada si activas esto:
                    # cv2.imshow("Máscara", mask_np)

    # Mostrar mensajes principales
    cv2.putText(frame, mensaje_calibracion, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0) if mm_per_pixel else (0, 0, 255), 2)

    if min_ancho_vastago is not None:
        min_redondeado = math.floor(min_ancho_vastago)
        cv2.putText(frame, f"MIN ACTUAL: {min_redondeado} mm", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    cv2.imshow("Medición Vástago", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
