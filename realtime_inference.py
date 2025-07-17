from ultralytics import YOLO
import cv2

# Cargar el modelo entrenado
model = YOLO("runs/segment/perno_seg_v12/weights/best.pt")  # Ajusta la ruta si usaste otro nombre

# Iniciar captura de la webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el frame.")
        break

    # Realizar predicción
    results = model.predict(source=frame, show=False, conf=0.3, task="segment")

    # Dibujar los resultados en el frame original
    annotated_frame = results[0].plot()

    # Mostrar el frame anotado
    cv2.imshow("Detección de Vástago (YOLOv8)", annotated_frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
