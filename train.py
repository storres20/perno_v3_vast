from ultralytics import YOLO

# Carga modelo base de segmentación
model = YOLO("yolov8s-seg.pt")  # yolov8m-seg.pt o yolov8s-seg.pt para mejor precisión

# Entrenar
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    task="segment",
    name="perno_seg_v1"
)
