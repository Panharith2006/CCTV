from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt", classes=None):
        """
        model_path: pre-trained YOLO model
        classes: list of class names to detect, e.g. ['person', 'mask', 'helmet']
        """
        self.model = YOLO(model_path)
        self.classes = classes

    def detect(self, frame):
        """
        Detect objects in a single frame.
        Returns list of detections.
        """
        results = self.model(frame)  # returns list of objects
        detections = []

        for res in results:
            for box in res.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = self.model.names[cls_id]

                if self.classes and class_name not in self.classes:
                    continue

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "class": class_name,
                    "confidence": conf
                })

        return detections
