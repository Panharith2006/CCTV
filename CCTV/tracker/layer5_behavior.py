class BehaviorDecider:
    def __init__(self, motion_threshold=50, loitering_warning_frames=360, loitering_alert_frames=720, warning_time=100, alert_time=200):
        """
        REVISED BEHAVIOR DECISION ENGINE
        
        Priority Rules (as specified):
        1. Helmet detected → ALERT
        2. Mask detected → WARNING
        3. Helmet + erratic motion → ALERT
        4. Mask + erratic motion → WARNING
        5. Erratic motion only → WARNING
        6. Standing still 6 min (360 frames @ 1fps) → WARNING
        7. Standing still 12 min (720 frames @ 1fps) → ALERT
        8. Normal movement → NORMAL
        
        Args:
            motion_threshold: pixels for erratic motion detection
            loitering_warning_frames: frames for loitering warning (6 min @ 1fps = 360)
            loitering_alert_frames: frames for loitering alert (12 min @ 1fps = 720)
        """
        self.motion_threshold = motion_threshold
        self.loitering_warning_frames = loitering_warning_frames
        self.loitering_alert_frames = loitering_alert_frames
        self.track_motion_history = {}  # track_id -> list of motion gaps
        self.track_low_motion_count = {}  # track_id -> count of low motion frames

    def update(self, tracks, motion_info):
        decisions = []

        for t, m in zip(tracks, motion_info):
            track_id = t["track_id"]
            attributes = t.get("attributes", {"mask": False, "helmet": False})

            # Save motion history
            if track_id not in self.track_motion_history:
                self.track_motion_history[track_id] = []
                self.track_low_motion_count[track_id] = 0
            
            current_motion = max(m["motion_gaps"].values()) if m["motion_gaps"] else 0
            self.track_motion_history[track_id].append(current_motion)
            
            # Count low motion frames (for loitering detection)
            if current_motion < 10:  # Low motion threshold
                self.track_low_motion_count[track_id] += 1
            else:
                self.track_low_motion_count[track_id] = 0  # Reset if moving

            # Get max motion from recent history
            history_window = 10
            recent_motion = self.track_motion_history[track_id][-history_window:]
            max_recent_motion = max(recent_motion) if recent_motion else 0
            
            # Get loitering frame count
            low_motion_frames = self.track_low_motion_count[track_id]

            # ============ REVISED DECISION ENGINE (PRIORITY RULES) ============
            decision = "Normal"
            reason = ""
            
            has_mask = attributes.get("mask", False)
            has_helmet = attributes.get("helmet", False)
            is_erratic = max_recent_motion > self.motion_threshold
            loitering_warning = low_motion_frames >= self.loitering_warning_frames
            loitering_alert = low_motion_frames >= self.loitering_alert_frames

            # Priority 1: Helmet = ALERT (highest priority)
            if has_helmet:
                if is_erratic:
                    decision = "Alert"
                    reason = "HELMET + Erratic motion"
                else:
                    decision = "Alert"
                    reason = "HELMET detected (identity concealment)"
            
            # Priority 2: Mask = WARNING
            elif has_mask:
                if is_erratic:
                    decision = "Warning"
                    reason = "MASK + Erratic motion"
                else:
                    decision = "Warning"
                    reason = "MASK detected (identity concealment)"
            
            # Priority 3: Loitering (standing still)
            elif loitering_alert:
                decision = "Alert"
                reason = f"Loitering {low_motion_frames//60:.1f} min (>12 min)"
            elif loitering_warning:
                decision = "Warning"
                reason = f"Loitering {low_motion_frames//60:.1f} min (>6 min)"
            
            # Priority 4: Erratic motion only
            elif is_erratic:
                decision = "Warning"
                reason = f"Erratic motion detected ({max_recent_motion:.0f} px)"
            
            # Otherwise: Normal
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
    