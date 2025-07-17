from ultralytics import YOLO
import cv2
import numpy as np

# Cargar modelo entrenado
model = YOLO("runs/segment/perno_seg_v12/weights/best.pt")

# Iniciar cámara
cap = cv2.VideoCapture(0)

# Cuadro de referencia real (en mm)
cuadro_real_mm = 23.0

def detectar_cuadro_referencia(image):
    """Detecta el cuadro negro de referencia (23 mm)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contornos:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            aspecto = w / float(h)
            if 0.9 < aspecto < 1.1 and 20 < w < 200:
                return w  # ancho en píxeles
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Paso 1: Calcular escala dinámica (mm/px)
    ancho_cuadro_px = detectar_cuadro_referencia(frame)
    if ancho_cuadro_px:
        mm_per_pixel = cuadro_real_mm / ancho_cuadro_px
        cv2.putText(frame, f"Escala: {mm_per_pixel:.2f} mm/px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        mm_per_pixel = None
        cv2.putText(frame, "Cuadro de 23mm NO detectado", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Paso 2: Segmentación
    results = model.predict(source=frame, conf=0.4, task="segment", show=False)[0]

    if results.masks:
        for mask in results.masks.data:
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                # Cortar la parte superior del contorno (donde suele estar la cabeza)
                x, y, w, h = cv2.boundingRect(cnt)
                altura_corte = int(h * 0.4)  # Cortar el 40% superior (ajustable)
                mascara_vastago = mask_np[y + altura_corte:y + h, x:x + w]

                # Buscar el ancho máximo real en la zona del vástago
                if mascara_vastago.size == 0:
                    continue

                proyeccion_horizontal = np.sum(mascara_vastago > 0, axis=0)
                ancho_max_px = np.max(proyeccion_horizontal)

                # Dibujar contorno ajustado
                cv2.rectangle(frame, (x, y + altura_corte), (x + w, y + h), (255, 255, 0), 2)

                if mm_per_pixel and ancho_max_px > 0:
                    ancho_mm = ancho_max_px * mm_per_pixel
                    cv2.putText(frame, f"Ancho: {ancho_mm:.2f} mm", (x, y + altura_corte - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Detección y Medición", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
