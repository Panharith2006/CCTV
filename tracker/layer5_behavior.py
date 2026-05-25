class BehaviorDecider:
    def __init__(self, motion_threshold=50, loitering_frames=10):
        """
        motion_threshold: pixels for alert
        loitering_frames: min frames of low motion for warning
        """
        self.motion_threshold = motion_threshold
        self.loitering_frames = loitering_frames
        self.track_motion_history = {}  # track_id -> list of motion gaps

    def update(self, tracks, motion_info):
        decisions = []

        for t, m in zip(tracks, motion_info):
            track_id = t["track_id"]
            attributes = t.get("attributes", {"mask": False, "helmet": False})

            # Save motion history
            if track_id not in self.track_motion_history:
                self.track_motion_history[track_id] = []
            self.track_motion_history[track_id].append(max(m["motion_gaps"].values()))

            max_gap = max(self.track_motion_history[track_id][-self.loitering_frames:])

            # Rule-based decision
            decision = "Normal"
            reason = ""

            if max_gap > self.motion_threshold:
                decision = "Alert"
                reason = "High motion detected (fleeing/falling/erratic)"
            elif attributes.get("mask") or attributes.get("helmet") or max_gap > self.motion_threshold/3:
                decision = "Warning"
                reason = "Mask/helmet detected or loitering"
            else:
                decision = "Normal"
                reason = "Minimal movement"

            decisions.append({
                "track_id": track_id,
                "frame_id": m["frame_id"],
                "decision": decision,
                "reason": reason
            })
            
        

        return decisions
    