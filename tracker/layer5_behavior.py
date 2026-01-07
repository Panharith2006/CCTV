class BehaviorDecider:
    def __init__(self, motion_threshold=50, loitering_frames=10, warning_time=100, alert_time=200):
        """
        motion_threshold: pixels for high motion alert
        loitering_frames: min frames of low motion
        warning_time: seconds before warning (not used in simple rules)
        alert_time: seconds before alert (not used in simple rules)
        """
        self.motion_threshold = motion_threshold
        self.loitering_frames = loitering_frames
        self.warning_time = warning_time
        self.alert_time = alert_time
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

            # ============ LAYER 8 — DECISION ENGINE (RULE-BASED) ============
            # PRIMARY RULE: Mask or Helmet = ABNORMAL (Warning/Alert)
            # Secondary rules: high motion, loitering
            decision = "Normal"
            reason = ""

            # Priority 1: Mask/Helmet (ABNORMAL - most important)
            if attributes.get("mask") or attributes.get("helmet"):
                # If wearing mask/helmet AND moving erratically = ALERT
                if max_gap > self.motion_threshold:
                    decision = "Alert"
                    reason = "ABNORMAL: Mask/Helmet + Erratic movement"
                else:
                    decision = "Warning"
                    reason = "ABNORMAL: Mask/Helmet detected"
            # Priority 2: Motion-based rules (only if no mask/helmet)
            elif max_gap > self.motion_threshold:
                decision = "Alert"
                reason = "Erratic motion detected"
            elif len(self.track_motion_history[track_id]) >= self.loitering_frames and max_gap < 10:
                decision = "Warning"
                reason = "Loitering detected"
            else:
                decision = "Normal"
                reason = "Normal behavior"

            decisions.append({
                "track_id": track_id,
                "frame_id": m["frame_id"],
                "decision": decision,
                "reason": reason
            })
            
        

        return decisions
    