from ultralytics import YOLO
import cv2
import numpy as np

# Cargar modelo segmentado
model = YOLO("runs/segment/perno_seg_v12/weights/best.pt")

# Inicializar webcam
cap = cv2.VideoCapture(0)

# Medida real del cuadro de referencia en mm
cuadro_real_mm = 23.0

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

    # Paso 1: Escala a partir del cuadro de 23 mm
    ancho_cuadro_px = detectar_cuadro_referencia(frame)
    if ancho_cuadro_px:
        mm_per_pixel = cuadro_real_mm / ancho_cuadro_px
        cv2.putText(frame, f"Escala: {mm_per_pixel:.2f} mm/px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        mm_per_pixel = None
        cv2.putText(frame, "Cuadro de 23mm NO detectado", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Paso 2: Segmentación YOLOv8
    results = model.predict(source=frame, conf=0.3, task="segment", show=False)[0]

    for mask in results.masks.data if results.masks else []:
        mask_np = mask.cpu().numpy().astype(np.uint8) * 255

        # Filtro morfológico
        kernel = np.ones((3, 3), np.uint8)
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)

        # Encontrar contornos
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # ===== Recortar zona inferior del contorno (eliminamos cabeza) =====
            corte_y = int(y + 0.4 * h)  # usamos el 60% inferior del contorno
            vastago_mask = mask_np[corte_y:y + h, x:x + w]

            # Buscar nuevo contorno en la parte recortada
            sub_contours, _ = cv2.findContours(vastago_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for scnt in sub_contours:
                sx, sy, sw, sh = cv2.boundingRect(scnt)
                sx += x
                sy += corte_y

                # Dibujar y medir
                cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (255, 255, 0), 2)
                if mm_per_pixel:
                    ancho_mm = sw * mm_per_pixel
                    cv2.putText(frame, f"Ancho: {ancho_mm:.2f} mm", (sx, sy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Detección y Medición", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
