import requests
import cv2
import os
from datetime import datetime

class TelegramNotifier:
    def __init__(self, bot_token, chat_id, save_snapshots=True):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.save_snapshots = save_snapshots
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
        self.tmp_dir = "snapshots"
        if self.save_snapshots and not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def send_alert(self, frame, track_id, cam_id, decision, reason, person_id=None, is_reidentified=False, violation_type=None):
        """
        Send enhanced alert with full context
        
        REVISED: Now includes violation type, new/re-identified status
        
        Args:
            frame: Video frame
            track_id: Track ID
            cam_id: Camera ID
            decision: WARNING or ALERT
            reason: Detailed reason
            person_id: Database person ID (if violation) or memory ID (if normal)
            is_reidentified: True if this is a re-identified suspect
            violation_type: MASK, HELMET, ERRATIC_MOTION, LOITERING, etc.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.tmp_dir}/cam{cam_id}_track{track_id}_{timestamp}.jpg"
        
        if self.save_snapshots:
            cv2.imwrite(filename, frame)
        else:
            filename = frame  # use frame directly if using memory buffer (advanced)

        # Build enhanced caption (Windows-compatible, no emoji to avoid encoding issues)
        caption_lines = [
            f"[{decision.upper()} ALERT]",
            "",
            f"Camera: {cam_id}",
            f"Track ID: {track_id}",
        ]
        
        # Add person ID and status
        if person_id is not None:
            if isinstance(person_id, int):
                # Database person (violation)
                if is_reidentified:
                    caption_lines.append(f"Person ID: {person_id} [RE-IDENTIFIED]")
                    caption_lines.append(f"WARNING: Known violator returned!")
                else:
                    caption_lines.append(f"Person ID: {person_id} [NEW SUSPECT]")
                    caption_lines.append(f"First time violation detected")
            else:
                # Memory person (shouldn't trigger alerts, but just in case)
                caption_lines.append(f"Temp ID: {person_id} (Not in database)")
        
        # Add violation type if available
        if violation_type:
            caption_lines.append(f"Violation Type: {violation_type}")
        
        caption_lines.extend([
            "",
            f"Reason: {reason}",
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ])
        
        caption = "\n".join(caption_lines)

        with open(filename, "rb") as img_file:
            files = {"photo": img_file}
            data = {"chat_id": self.chat_id, "caption": caption}
            try:
                response = requests.post(self.base_url, files=files, data=data, timeout=10)
                if response.status_code == 200:
                    print(f"[Telegram] [OK] Alert sent successfully")
                else:
                    print(f"[Telegram] [X] Failed to send alert: {response.status_code}")
            except Exception as e:
                print(f"[Telegram] [X] Failed to send alert: {e}")
