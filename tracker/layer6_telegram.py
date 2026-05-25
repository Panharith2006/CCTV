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

    def send_alert(self, frame, track_id, cam_id, decision, reason):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.tmp_dir}/cam{cam_id}_track{track_id}_{timestamp}.jpg"
        
        if self.save_snapshots:
            cv2.imwrite(filename, frame)
        else:
            filename = frame  # use frame directly if using memory buffer (advanced)

        caption = f"Camera: {cam_id}\nTrack ID: {track_id}\nDecision: {decision}\nReason: {reason}"

        with open(filename, "rb") as img_file:
            files = {"photo": img_file}
            data = {"chat_id": self.chat_id, "caption": caption}
            try:
                requests.post(self.base_url, files=files, data=data, timeout=5)
            except Exception as e:
                print(f"[Telegram] Failed to send alert: {e}")
