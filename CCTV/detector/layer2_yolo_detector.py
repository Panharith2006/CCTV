from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path="yolov8n.pt", classes=None, conf_threshold=0.6):
        """
        model_path: pre-trained YOLO model or checkpoint path
        classes: list of class names to detect, e.g. ['person', 'mask', 'helmet']
        conf_threshold: minimum confidence to keep a detection
        """
        self.model = YOLO(model_path)
        self.classes = classes
        self.conf_threshold = conf_threshold

        # Debug: print loaded model class names
        try:
            print(f"[YOLO] Loaded model: {model_path}; classes: {self.model.names}")
        except Exception:
            print(f"[YOLO] Loaded model: {model_path}; (could not read names)")

    def detect(self, frame):
        """
        Detect objects in a single frame. Returns list of detections filtered by class and confidence.
        """
        results = self.model(frame)
        detections = []

        for res in results:
            # Debug: how many boxes were produced before filtering
            try:
                boxes = list(res.boxes)
                print(f"[YOLO] raw boxes: {len(boxes)}")
            except Exception:
                boxes = []
                print("[YOLO] raw boxes: 0 (no boxes attribute)")

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = self.model.names.get(cls_id, str(cls_id)) if hasattr(self.model, 'names') else str(cls_id)

                print(f"[YOLO] box -> class_id={cls_id} name={class_name} conf={conf:.3f} bbox={[x1,y1,x2,y2]}")

                if self.classes and class_name not in self.classes:
                    continue

                if conf < self.conf_threshold:
                    continue

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "class": class_name,
                    "confidence": conf
                })

        return detections
