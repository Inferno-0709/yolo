# src/detection/yolo_pipeline.py
from ultralytics import YOLO
import torch
import json

class YOLODetector:
    def __init__(self, model_path="yolo11x.pt"):
        """
        Initialize YOLO model.
        - Loads the latest (larger) YOLOv8x model.
        - Automatically uses GPU if available, else CPU.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {self.device.upper()}")

        # Load the YOLOv8x model (most accurate among YOLOv8 family)
        self.model = YOLO(model_path)
        self.model.to(self.device)  # Move model to GPU if available

    def detect_objects(self, source):
        """
        Runs YOLO object detection and returns structured output.
        """
        # Run detection on the selected device
        results = self.model(source, device=self.device)

        structured_output = []
        for result in results:
            for box in result.boxes:
                cls = self.model.names[int(box.cls)]
                conf = float(box.conf)
                xyxy = box.xyxy[0].tolist()
                structured_output.append({
                    "object": cls,
                    "confidence": conf,
                    "bounding_box": xyxy
                })

        return structured_output


if __name__ == "__main__":
    detector = YOLODetector()
    image_path = "data/images/sample.jpg"
    detections = detector.detect_objects(image_path)
    print(json.dumps(detections, indent=2))
