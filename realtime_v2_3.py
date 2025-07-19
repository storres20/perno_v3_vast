from ultralytics import YOLO
import cv2
import numpy as np
import math

model = YOLO("runs/segment/perno_seg_v12/weights/best.pt")
cap = cv2.VideoCapture(1)

cuadro_real_mm = 23.0
min_ancho_vastago = None  # mínimo ancho actual (solo mientras el vástago está en escena)

def detectar_cuadro_referencia(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contornos:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            aspecto = w / float(h)
            if 0.9 < aspecto < 1.1 and 20 < w < 200:
                return w
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ancho_cuadro_px = detectar_cuadro_referencia(frame)
    if ancho_cuadro_px:
        mm_per_pixel = cuadro_real_mm / ancho_cuadro_px
        cv2.putText(frame, f"Escala: {mm_per_pixel:.2f} mm/px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        mm_per_pixel = None
        cv2.putText(frame, "Cuadro de 23mm NO detectado", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    results = model.predict(source=frame, conf=0.3, task="segment", show=False)[0]
    masks = results.masks.data if results.masks else []

    if len(masks) == 0:
        min_ancho_vastago = None

    for mask in masks:
        mask_np = mask.cpu().numpy().astype(np.uint8) * 255

        # Filtro morfológico para evitar ruido
        kernel = np.ones((3, 3), np.uint8)
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Cortar solo la parte del vástago (sin cabeza)
            corte_y = int(y + 0.4 * h)
            vastago_mask = mask_np[corte_y:y + h, x:x + w]

            sub_contours, _ = cv2.findContours(vastago_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for scnt in sub_contours:
                sx, sy, sw, sh = cv2.boundingRect(scnt)
                sx += x
                sy += corte_y

                if mm_per_pixel:
                    ancho_mm = sw * mm_per_pixel

                    # Actualizar mínimo si es necesario
                    if ancho_mm > 2.5:  # Filtra falsos positivos por ruido
                        if min_ancho_vastago is None or ancho_mm < min_ancho_vastago:
                            min_ancho_vastago = ancho_mm

                    # Dibujar medición actual
                    cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (255, 255, 0), 2)
                    cv2.putText(frame, f"Ancho: {ancho_mm:.2f} mm", (sx, sy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Mostrar el mínimo del vástago actual (si hay alguno)
    if min_ancho_vastago is not None:
        min_redondeado = math.floor(min_ancho_vastago)
        cv2.putText(frame, f"MIN ACTUAL: {min_redondeado} mm", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Medición Vástago", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
