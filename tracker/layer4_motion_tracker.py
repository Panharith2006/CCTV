import numpy as np
from collections import defaultdict

class MotionAnalyzer:
    def __init__(self, frame_gaps=[1,5,10,15,20]):
        self.frame_gaps = frame_gaps
        self.track_history = defaultdict(list)  # track_id -> list of centers

    def update(self, tracks, frame_id):
        """
        tracks: list of dicts from SORT
        frame_id: current frame index
        """
        motion_info = []

        for t in tracks:
            track_id = t["track_id"]
            bbox = t["bbox"]
            x1, y1, x2, y2 = bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            self.track_history[track_id].append((frame_id, cx, cy))

            motion_gaps = {}
            for gap in self.frame_gaps:
                if len(self.track_history[track_id]) > gap:
                    old_frame_id, old_cx, old_cy = self.track_history[track_id][-gap-1]
                    distance = np.sqrt((cx - old_cx)**2 + (cy - old_cy)**2)
                    motion_gaps[gap] = distance
                else:
                    motion_gaps[gap] = 0.0

            motion_info.append({
                "track_id": track_id,
                "frame_id": frame_id,
                "motion_gaps": motion_gaps
            })

        return motion_info
