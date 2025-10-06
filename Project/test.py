from ultralytics import YOLO

model = YOLO("yolo11x.pt")
print(model.names)
