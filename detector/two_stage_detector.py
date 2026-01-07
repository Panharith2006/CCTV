from ultralytics import YOLO

class TwoStageDetector:
    """Two-stage detector:
    - Stage 1: person detection using COCO-pretrained model (yolov8n.pt)
    - Stage 2: attribute detection using a fine-tuned model (mask/helmet)

    Returns unified list of detections with keys: bbox, class, confidence
    """

    def __init__(self, person_model_path="yolov8n.pt", attr_model_path=None, conf_person=0.5, conf_attr=0.2):
        self.person_model = YOLO(person_model_path)
        self.attr_model = YOLO(attr_model_path) if attr_model_path is not None else None
        self.conf_person = conf_person
        self.conf_attr = conf_attr

        print(f"[TwoStage] person_model classes: {self.person_model.names}")
        if self.attr_model is not None:
            try:
                print(f"[TwoStage] attr_model classes: {self.attr_model.names}")
            except Exception:
                print(f"[TwoStage] Loaded attr model: {attr_model_path}")

    def detect(self, frame, imgsz=416):
        detections = []

        # Stage 1: persons
        res_p = self.person_model(frame, imgsz=imgsz)
        for r in res_p:
            boxes = list(r.boxes)
            for b in boxes:
                cls = int(b.cls[0])
                name = self.person_model.names.get(cls, str(cls)) if hasattr(self.person_model, 'names') else str(cls)
                conf = float(b.conf[0])
                if name == 'person' and conf >= self.conf_person:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    detections.append({"bbox": [x1, y1, x2, y2], "class": 'person', "confidence": conf})

        # Run attribute detector on each person's head crop for better small-object detection
        if self.attr_model is not None and len(detections) > 0:
            for person in detections[:]:
                px1, py1, px2, py2 = person["bbox"]
                # define head region (top fraction of person bbox)
                head_h = max(1, int((py2 - py1) * 0.45))
                hx1, hy1, hx2, hy2 = px1, py1, px2, py1 + head_h

                # clip to image
                ih, iw = frame.shape[:2]
                hx1 = max(0, hx1); hy1 = max(0, hy1); hx2 = min(iw, hx2); hy2 = min(ih, hy2)

                if hx2 <= hx1 or hy2 <= hy1:
                    continue

                crop = frame[hy1:hy2, hx1:hx2]
                # run attr model on the crop
                res_a = self.attr_model(crop, imgsz=imgsz)
                found = False
                for r in res_a:
                    boxes = list(r.boxes)
                    # Debug: how many attr boxes on this crop
                    print(f"[TwoStage] attr raw boxes on head crop: {len(boxes)} for person bbox {person['bbox']}")
                    for b in boxes:
                        cls = int(b.cls[0])
                        name = self.attr_model.names.get(cls, str(cls)) if hasattr(self.attr_model, 'names') else str(cls)
                        conf = float(b.conf[0])
                        if conf < self.conf_attr:
                            continue
                        lname = name.lower()
                        if 'mask' in lname:
                            label = 'mask'
                        elif 'helmet' in lname or 'hardhat' in lname:
                            label = 'helmet'
                        else:
                            label = lname

                        # map crop box coords back to full frame coordinates
                        x1c, y1c, x2c, y2c = map(int, b.xyxy[0])
                        # ensure coordinates are within crop
                        x1_full = hx1 + max(0, x1c)
                        y1_full = hy1 + max(0, y1c)
                        x2_full = hx1 + min(x2c, hx2-hx1)
                        y2_full = hy1 + min(y2c, hy2-hy1)

                        detections.append({"bbox": [x1_full, y1_full, x2_full, y2_full], "class": label, "confidence": conf})
                        found = True
                if not found:
                    # Fallback: run attribute detector on full person bbox if head crop had no attr
                    # This helps when the mask detector needs more context.
                    px1, py1, px2, py2 = person["bbox"]
                    ih, iw = frame.shape[:2]
                    fx1 = max(0, px1); fy1 = max(0, py1); fx2 = min(iw, px2); fy2 = min(ih, py2)
                    if fx2 > fx1 and fy2 > fy1:
                        full_crop = frame[fy1:fy2, fx1:fx2]
                        res_af = self.attr_model(full_crop, imgsz=imgsz)
                        for r2 in res_af:
                            boxes2 = list(r2.boxes)
                            print(f"[TwoStage] attr raw boxes on full person crop: {len(boxes2)} for person bbox {person['bbox']}")
                            for b2 in boxes2:
                                cls2 = int(b2.cls[0])
                                name2 = self.attr_model.names.get(cls2, str(cls2)) if hasattr(self.attr_model, 'names') else str(cls2)
                                conf2 = float(b2.conf[0])
                                if conf2 < self.conf_attr:
                                    continue
                                lname2 = name2.lower()
                                if 'mask' in lname2:
                                    label2 = 'mask'
                                elif 'helmet' in lname2 or 'hardhat' in lname2:
                                    label2 = 'helmet'
                                else:
                                    label2 = lname2

                                x1c2, y1c2, x2c2, y2c2 = map(int, b2.xyxy[0])
                                x1_full2 = fx1 + max(0, x1c2)
                                y1_full2 = fy1 + max(0, y1c2)
                                x2_full2 = fx1 + min(x2c2, fx2-fx1)
                                y2_full2 = fy1 + min(y2c2, fy2-fy1)

                                detections.append({"bbox": [x1_full2, y1_full2, x2_full2, y2_full2], "class": label2, "confidence": conf2})

        return detections
