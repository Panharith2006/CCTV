from ultralytics import YOLO

class SingleStageDetector:
    """Targeted detection workflow:
    1. Detect persons on full frame
    2. Extract head region from each person
    3. Detect mask/helmet on head crops
    
    Uses a single trained model (e.g., best.pt) for both stages.
    
    Detection Priority:
    - Helmet → Alert
    - Mask → Warning
    
    Returns unified list of detections with keys: bbox, class, confidence
    """

    def __init__(self, model_path="best.pt", conf_person=0.40, conf_mask=0.50, conf_helmet=0.65, head_fraction=0.45):
        """
        Initialize detector with trained model.
        
        Args:
            model_path: Path to trained YOLO model with person, mask, helmet classes
            conf_person: Confidence threshold for person detection (LOWERED to 0.40 for better detection)
            conf_mask: Confidence threshold for mask detection (0.50 - balanced)
            conf_helmet: Confidence threshold for helmet detection (HIGHER 0.65 - filter false positives)
            head_fraction: Fraction of person bbox to use as head region (0.45 = top 45%)
        
        Threshold Reasoning:
        - Person (0.40): Lower = detect more people (safer for security)
        - Helmet (0.65): Higher = filter false helmet detections due to model inaccuracy
            Why? Model may falsely detect hair/hoods/caps as helmets (false positives)
            Higher threshold = only accept confident detections = prevents missed violations
            Lower threshold would accept weak false detections = person without helmet seen as "has helmet"
        - Mask (0.50): Balanced threshold for mask detection (WARNING level)
        """
        self.model = YOLO(model_path)
        self.conf_person = conf_person
        self.conf_mask = conf_mask
        self.conf_helmet = conf_helmet
        self.head_fraction = head_fraction
        
        # Validate model has required classes
        if hasattr(self.model, 'names'):
            model_classes = [str(name).lower() for name in self.model.names.values()]
            has_person = 'person' in model_classes
            has_mask = 'mask' in model_classes
            has_helmet = 'helmet' in model_classes
            
            if not (has_person and has_mask and has_helmet):
                print(f"[WARNING] Model may be missing required classes!")
                print(f"[WARNING] Expected: person, mask, helmet")
                print(f"[WARNING] Found: {self.model.names}")
            else:
                print(f"[Detector] Targeted detection workflow enabled")
                print(f"[Detector] Model: {model_path}")
                print(f"[Detector] Classes: {self.model.names}")
                print(f"[Detector] Strategy: Person detection → Head crop → Mask/Helmet detection")
                print(f"[Detector] Thresholds: person={conf_person}, mask={conf_mask}, helmet={conf_helmet}")
        else:
            print(f"[Detector] Model loaded: {model_path}")

    def detect(self, frame, imgsz=640):
        """
        Targeted workflow: Detect persons, then detect mask/helmet on head crops.
        
        Args:
            frame: Input frame (numpy array)
            imgsz: Input size for YOLO inference
            
        Returns:
            List of detections with keys: bbox, class, confidence
        """
        detections = []
        
        # STEP 1: Detect persons on full frame
        persons = self._detect_persons(frame, imgsz)
        detections.extend(persons)
        
        # STEP 2: For each person, extract head region and detect mask/helmet
        if len(persons) > 0:
            attributes = self._detect_attributes_on_heads(frame, persons, imgsz)
            detections.extend(attributes)
        
        return detections
    
    def _detect_persons(self, frame, imgsz):
        """Step 1: Detect persons on full frame"""
        persons = []
        
        results = self.model(frame, imgsz=imgsz, verbose=False)
        
        for r in results:
            boxes = list(r.boxes)
            for b in boxes:
                cls = int(b.cls[0])
                name = self.model.names.get(cls, str(cls))
                conf = float(b.conf[0])
                
                # Only extract persons
                if name.lower() == 'person' and conf >= self.conf_person:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    persons.append({
                        "bbox": [x1, y1, x2, y2], 
                        "class": 'person', 
                        "confidence": conf
                    })
        
        return persons
    
    def _detect_attributes_on_heads(self, frame, persons, imgsz):
        """Step 2: Detect mask/helmet on head regions of detected persons"""
        attributes = []
        ih, iw = frame.shape[:2]
        
        for person in persons:
            px1, py1, px2, py2 = person["bbox"]
            
            # Define head region (top fraction of person bbox)
            head_h = max(1, int((py2 - py1) * self.head_fraction))
            hx1, hy1, hx2, hy2 = px1, py1, px2, py1 + head_h
            
            # Clip to image boundaries
            hx1 = max(0, hx1)
            hy1 = max(0, hy1)
            hx2 = min(iw, hx2)
            hy2 = min(ih, hy2)
            
            if hx2 <= hx1 or hy2 <= hy1:
                continue
            
            # Extract head crop
            head_crop = frame[hy1:hy2, hx1:hx2]
            
            # Run detection on head crop
            results = self.model(head_crop, imgsz=imgsz, verbose=False)
            
            for r in results:
                boxes = list(r.boxes)
                for b in boxes:
                    cls = int(b.cls[0])
                    name = self.model.names.get(cls, str(cls))
                    conf = float(b.conf[0])
                    lname = name.lower()
                    
                    # Only process mask and helmet detections
                    if lname in ['mask', 'with-mask', 'wearing-mask', 'has-mask']:
                        if conf >= self.conf_mask:  # Use mask-specific threshold
                            # Map crop coordinates back to full frame
                            x1c, y1c, x2c, y2c = map(int, b.xyxy[0])
                            x1_full = hx1 + max(0, x1c)
                            y1_full = hy1 + max(0, y1c)
                            x2_full = hx1 + min(x2c, hx2 - hx1)
                            y2_full = hy1 + min(y2c, hy2 - hy1)
                            
                            attributes.append({
                                "bbox": [x1_full, y1_full, x2_full, y2_full],
                                "class": 'mask',
                                "confidence": conf
                            })
                    
                    elif lname in ['helmet', 'with-helmet', 'hardhat', 'wearing-helmet', 'has-helmet']:
                        if conf >= self.conf_helmet:  # Use helmet-specific threshold (lower for fewer false alerts)
                            # Map crop coordinates back to full frame
                            x1c, y1c, x2c, y2c = map(int, b.xyxy[0])
                            x1_full = hx1 + max(0, x1c)
                            y1_full = hy1 + max(0, y1c)
                            x2_full = hx1 + min(x2c, hx2 - hx1)
                            y2_full = hy1 + min(y2c, hy2 - hy1)
                            
                            attributes.append({
                                "bbox": [x1_full, y1_full, x2_full, y2_full],
                                "class": 'helmet',
                                "confidence": conf
                            })
        
        return attributes
