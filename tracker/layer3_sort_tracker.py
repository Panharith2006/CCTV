import numpy as np
from sort.sort import Sort  # you can vendor SORT or install a package

class SortTracker:
    def __init__(self):
        self.tracker = Sort(
            max_age=30,
            min_hits=3,
            iou_threshold=0.3
        )

    def update(self, detections):
        """
        detections: list of dicts with bbox + confidence
        Returns tracked objects with IDs
        """
        if len(detections) == 0:
            dets = np.empty((0, 5))
        else:
            dets = np.array([
                [
                    d["bbox"][0],
                    d["bbox"][1],
                    d["bbox"][2],
                    d["bbox"][3],
                    d["confidence"]
                ]
                for d in detections if d["class"] == "person"
            ])

        tracks = self.tracker.update(dets)

        results = []
        for t in tracks:
            x1, y1, x2, y2, track_id = map(int, t)
            results.append({
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2],
                "class": "person"
            })

        return results
