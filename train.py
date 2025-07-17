from ultralytics import YOLO

# Cargar modelo base de segmentación
model = YOLO("yolov8s-seg.pt")  # Usa yolov8m-seg.pt o yolov8s-seg.pt si quieres mejor precisión

# Entrenar
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    task="segment",  # opcional en versiones recientes, pero puedes dejarlo
    name="perno_seg_v1"  # opcional, para organizar mejor tus runs
)
