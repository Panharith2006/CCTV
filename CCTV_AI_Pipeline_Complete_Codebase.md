# CCTV AI Pipeline - Complete Codebase

## Project Structure

```
cctv_ai/
├── config/
│   ├── layer0_cameras.py
│   └── telegram_config.py
├── ingest/
│   └── layer1_frame_ingest.py
├── detector/
│   └── layer2_yolo_detector.py
├── tracker/
│   ├── layer3_sort_tracker.py
│   ├── layer4_motion_tracker.py
│   ├── layer5_behavior.py
│   └── layer6_telegram.py
├── sort/
│   └── sort.py
├── test/
│   ├── layer1.py
│   ├── layer2.py
│   ├── layer3.py
│   ├── layer4.py
│   ├── layer5.py
│   └── layer6.py
├── main.py
├── setup.md
└── yolov8n.pt
```

---

## Root Files

### main.py
**Purpose:** Main pipeline orchestrator that integrates all layers (L1-L6) with Telegram notifications

```python
import cv2
import time

from config.layer0_cameras import CAMERAS
from ingest.layer1_frame_ingest import FrameIngestor
from detector.layer2_yolo_detector import YOLODetector
from tracker.layer3_sort_tracker import SortTracker
from tracker.layer4_motion_tracker import MotionAnalyzer
from tracker.layer5_behavior import BehaviorDecider
from tracker.layer6_telegram import TelegramNotifier
from config.telegram_config import BOT_TOKEN, CHAT_ID


# ===================== TELEGRAM COOLDOWN =====================
last_telegram_sent = {}  # (track_id, decision) -> timestamp


def can_send(track_id, decision, cooldown):
    now = time.time()
    key = (track_id, decision)

    if key not in last_telegram_sent:
        last_telegram_sent[key] = now
        return True

    if now - last_telegram_sent[key] >= cooldown:
        last_telegram_sent[key] = now
        return True

    return False


# ===================== UTIL =====================
def check_bbox_overlap(person_bbox, ppe_bbox):
    px1, py1, px2, py2 = person_bbox
    dx1, dy1, dx2, dy2 = ppe_bbox

    cx = (dx1 + dx2) / 2
    cy = (dy1 + dy2) / 2

    return px1 <= cx <= px2 and py1 <= cy <= py2


# ===================== MAIN =====================
def main():
    cam = CAMERAS[0]

    ingestor = FrameIngestor(
        camera_id=cam["camera_id"],
        source=cam["source"],
        sample_rate=3
    )

    detector = YOLODetector(classes=["person"])
    tracker = SortTracker()
    motion = MotionAnalyzer(frame_gaps=[1, 5, 10, 15, 20])

    behavior = BehaviorDecider(
        warning_time=100,   # warning at 100s
        alert_time=200      # alert at 200s
    )

    telegram = TelegramNotifier(BOT_TOKEN, CHAT_ID)

    print("[INFO] CCTV pipeline started")

    for data in ingestor.read():
        frame = data["frame"]
        frame_id = data["frame_id"]

        detections = detector.detect(frame)
        persons = [d for d in detections if d["class"] == "person"]

        tracks = tracker.update(persons, frame)

        motion_info = motion.update(tracks, frame_id)
        behavior_info = behavior.update(tracks, motion_info)

        for t, b in zip(tracks, behavior_info):
            track_id = t["track_id"]
            decision = b["decision"]
            reason = b["reason"]

            x1, y1, x2, y2 = t["bbox"]

            color = (255, 0, 0)
            if decision == "Warning":
                color = (0, 165, 255)
            elif decision == "Alert":
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"ID {track_id} | {decision}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            # -------- TELEGRAM LOGIC --------
            if decision == "Alert":
                if can_send(track_id, "Alert", cooldown=240):
                    telegram.send_alert(frame, track_id, cam["camera_id"], decision, reason)
                    print("[TELEGRAM] ALERT sent")

            elif decision == "Warning":
                if can_send(track_id, "Warning", cooldown=240):
                    telegram.send_alert(frame, track_id, cam["camera_id"], decision, reason)
                    print("[TELEGRAM] WARNING sent")

        cv2.imshow("CCTV AI", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    ingestor.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

### setup.md
**Purpose:** Installation dependencies

```markdown
pip install opencv-python

pip install ultralytics

pip install filterpy lap

pip install sort-tracker

pip install scikit-image
```

---

## config/ Folder

### layer0_cameras.py
**Purpose:** Camera configuration settings

```python
# config/cameras.py

CAMERAS = [
    #start with webcam first (comment on 2 other)
    {
        "camera_id": "CAM_01",
        "location": "Entrance",
        "source": "video/Standing_Still.mp4"
    }
    #{
    #   "camera_id": "CAM_02",
    #    "location": "Laptop Webcam",
    #    "source": 0
    #}
    #{
    #    "camera_id": "CAM_03",
    #    "location": "Real CCTV",
    #    "source": "rtsp://user:pass@ip:port/stream"
    #}
]
```

### telegram_config.py
**Purpose:** Telegram bot configuration

```python
BOT_TOKEN = "8377837060:AAF3ImKvApZn7BI8O8vxSfIddFek1YgZYHM"
CHAT_ID = "875628871"   # your user or group chat id
```

---

## ingest/ Folder

### layer1_frame_ingest.py
**Purpose:** Layer 1 - Frame ingestion from camera sources

```python
import cv2
import time

class FrameIngestor:
    def __init__(self, camera_id, source, sample_rate=3):
        self.camera_id = camera_id
        self.source = source
        self.sample_rate = sample_rate
        self.cap = cv2.VideoCapture(source)
        self.frame_count = 0

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open stream/video: {source}")

    def read(self):
        """
        Generator that yields sampled frames.
        Stops cleanly when video ends.
        """
        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("[INFO] Video stream ended")
                return   # ✅ IMPORTANT: stop generator

            self.frame_count += 1

            # Sampling
            if self.frame_count % self.sample_rate != 0:
                continue

            yield {
                "camera_id": self.camera_id,
                "timestamp": time.time(),
                "frame_id": self.frame_count,
                "frame": frame
            }

    def release(self):
        self.cap.release()
```

---

## detector/ Folder

### layer2_yolo_detector.py
**Purpose:** Layer 2 - YOLO object detection

```python
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt", classes=None):
        """
        model_path: pre-trained YOLO model
        classes: list of class names to detect, e.g. ['person', 'mask', 'helmet']
        """
        self.model = YOLO(model_path)
        self.classes = classes

    def detect(self, frame):
        """
        Detect objects in a single frame.
        Returns list of detections.
        """
        results = self.model(frame)  # returns list of objects
        detections = []

        for res in results:
            for box in res.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = self.model.names[cls_id]

                if self.classes and class_name not in self.classes:
                    continue

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "class": class_name,
                    "confidence": conf
                })

        return detections
```

---

## tracker/ Folder

### layer3_sort_tracker.py
**Purpose:** Layer 3 - DeepSort tracking algorithm wrapper

```python
from deep_sort_realtime.deepsort_tracker import DeepSort

class SortTracker:
    def __init__(self, max_age=30, n_init=3):
        # Initialize DeepSort tracker
        self.tracker = DeepSort(max_age=max_age, n_init=n_init)

    def update(self, detections, frame=None):
        """
        detections: list of dicts with keys ["bbox", "class", "confidence"]
        frame: current frame for embedding generation (required by DeepSort)
        """
        dets = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            conf = d.get("confidence", 1.0)
            dets.append(([x1, y1, x2, y2], conf, d["class"]))

        if frame is None:
            # DeepSort requires either embeddings or frame
            raise ValueError("Frame must be provided for tracking")

        tracks = self.tracker.update_tracks(dets, frame=frame)

        results = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            x1, y1, x2, y2 = t.to_ltrb()
            results.append({
                "track_id": t.track_id,
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })
        return results
```

### layer4_motion_tracker.py
**Purpose:** Layer 4 - Motion analysis with advanced metrics (gaps, direction changes, speed, displacement)

```python
import numpy as np
import math
from collections import defaultdict, deque

class MotionAnalyzer:
    def __init__(self, frame_gaps=[1,5,10,15,20], history_size=25):
        self.frame_gaps = frame_gaps
        self.history_size = history_size
        self.track_history = defaultdict(lambda: deque(maxlen=history_size))

    def _angle(self, p1, p2):
        return math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))

    def update(self, tracks, frame_id):
        motion_info = []

        for t in tracks:
            track_id = t["track_id"]
            x1, y1, x2, y2 = t["bbox"]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            self.track_history[track_id].append((frame_id, cx, cy))
            history = list(self.track_history[track_id])

            # --- Original motion gaps (KEEP) ---
            motion_gaps = {}
            for gap in self.frame_gaps:
                if len(history) > gap:
                    _, old_x, old_y = history[-gap-1]
                    motion_gaps[gap] = np.hypot(cx-old_x, cy-old_y)
                else:
                    motion_gaps[gap] = 0.0

            # --- New: direction changes ---
            angles = []
            for i in range(1, len(history)):
                angles.append(self._angle(history[i-1][1:], history[i][1:]))

            direction_changes = 0
            for i in range(1, len(angles)):
                if abs(angles[i] - angles[i-1]) > 45:
                    direction_changes += 1

            # --- New: speed ---
            total_dist = 0
            for i in range(1, len(history)):
                x0, y0 = history[i-1][1:]
                x1_, y1_ = history[i][1:]
                total_dist += math.hypot(x1_-x0, y1_-y0)

            total_time = max(len(history)-1, 1)
            avg_speed = total_dist / total_time

            # --- New: displacement ---
            displacement = math.hypot(
                history[-1][1] - history[0][1],
                history[-1][2] - history[0][2]
            )

            motion_info.append({
                "track_id": track_id,
                "frame_id": frame_id,
                "motion_gaps": motion_gaps,
                "direction_changes": direction_changes,
                "avg_speed": avg_speed,
                "displacement": displacement
            })

        return motion_info
```

### layer5_behavior.py
**Purpose:** Layer 5 - Behavior decision logic with time-based standing still detection and disoriented behavior

```python
import time
import math

class BehaviorDecider:
    def __init__(
        self,
        move_threshold=50,
        warning_time=100,
        alert_time=250,
        dis_dir_changes = 6,
        dis_speed = 30,
        dis_displacement = 150
    ):
        self.MOVE_THRESHOLD = move_threshold
        self.WARNING_TIME = warning_time
        self.ALERT_TIME = alert_time

        self.DIS_DIR = dis_dir_changes
        self.DIS_SPEED = dis_speed
        self.DIS_DIST = dis_displacement

        self.last_position = {}
        self.still_start_time = {}

    def _distance(self, p1, p2):
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

    def update(self, tracks, motion_info):
        now = time.time()
        results = []

        motion_map = {m["track_id"]: m for m in motion_info}

        for t in tracks:
            track_id = t["track_id"]
            x1, y1, x2, y2 = t["bbox"]
            attr = t.get("attributes", {})

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            pos = (cx, cy)

            # --- First time seen ---
            if track_id not in self.last_position:
                self.last_position[track_id] = pos
                self.still_start_time[track_id] = None   # IMPORTANT FIX
                results.append({
                    "track_id": track_id,
                    "decision": "Normal",
                    "reason": ""
                })
                continue

            dist = self._distance(self.last_position[track_id], pos)

            # --- MOVEMENT HANDLING (FIX) ---
            if dist >= self.MOVE_THRESHOLD:
                # Person moved → reset still timer
                self.still_start_time[track_id] = None
            else:
                # Person stopped → start timer if not started
                if self.still_start_time[track_id] is None:
                    self.still_start_time[track_id] = now

            still_time = (
                now - self.still_start_time[track_id]
                if self.still_start_time[track_id]
                else 0
            )

            decision = "Normal"
            reason = ""

            m = motion_map.get(track_id, {})

            # --- PPE (highest priority) ---
            if attr.get("helmet", False):
                decision = "Alert"
                reason = "Helmet worn"

            elif attr.get("mask", False) or attr.get("hat", False):
                decision = "Warning"
                reason = "Mask or hat worn"

            # --- Disoriented behavior ---
            elif (
                m.get("avg_speed", 0) > self.DIS_SPEED   # MUST be moving
                and m.get("direction_changes", 0) >= self.DIS_DIR
                and m.get("displacement", 0) <= self.DIS_DIST
            ):

                decision = "Warning"
                reason = "Disoriented navigation"

            # --- Standing still (FIXED) ---
            elif self.still_start_time[track_id] is not None:
                if still_time >= self.ALERT_TIME:
                    decision = "Alert"
                    reason = "Standing still too long"
                elif still_time >= self.WARNING_TIME:
                    decision = "Warning"
                    reason = "Standing still"

            self.last_position[track_id] = pos

            results.append({
                "track_id": track_id,
                "decision": decision,
                "reason": reason
            })

        return results
```

### layer6_telegram.py
**Purpose:** Layer 6 - Telegram notification system with cooldown

```python
import requests
import cv2
import os
from datetime import datetime, timedelta

class TelegramNotifier:
    def __init__(self, bot_token, chat_id, cooldown_sec=240, save_snapshots=True):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.cooldown_sec = cooldown_sec  # 4 min cooldown
        self.save_snapshots = save_snapshots
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
        self.tmp_dir = "snapshots"
        self.last_sent = {}  # track_id -> timestamp

        if self.save_snapshots and not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def send_alert(self, frame, track_id, cam_id, decision, reason):
        import time, cv2, requests
        now = time.time()
        last = self.last_sent.get(track_id, 0)

        if now - last < self.cooldown_sec:
            return  # skip sending, cooldown not reached

        self.last_sent[track_id] = now

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.tmp_dir}/cam{cam_id}_track{track_id}_{timestamp}.jpg"

        if self.save_snapshots:
            cv2.imwrite(filename, frame)
        else:
            filename = frame

        caption = f"Camera: {cam_id}\nTrack ID: {track_id}\nDecision: {decision}\nReason: {reason}"

        try:
            with open(filename, "rb") as img_file:
                files = {"photo": img_file}
                data = {"chat_id": self.chat_id, "caption": caption}
                requests.post(self.base_url, files=files, data=data, timeout=5)
        except Exception as e:
            print(f"[Telegram] Failed: {e}")
```

---

## sort/ Folder

### sort.py
**Purpose:** SORT (Simple Online and Realtime Tracking) algorithm implementation

```python
"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o)  


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  iou_matrix = iou_batch(detections, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 5))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
  # all train
  args = parse_args()
  display = args.display
  phase = args.phase
  total_time = 0.0
  total_frames = 0
  colours = np.random.rand(32, 3) #used only for display
  if(display):
    if not os.path.exists('mot_benchmark'):
      print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
      exit()
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')

  if not os.path.exists('output'):
    os.makedirs('output')
  pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
  for seq_dets_fn in glob.glob(pattern):
    mot_tracker = Sort(max_age=args.max_age, 
                       min_hits=args.min_hits,
                       iou_threshold=args.iou_threshold) #create instance of the SORT tracker
    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
    seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]
    
    with open(os.path.join('output', '%s.txt'%(seq)),'w') as out_file:
      print("Processing %s."%(seq))
      for frame in range(int(seq_dets[:,0].max())):
        frame += 1 #detection and frame numbers begin at 1
        dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
        dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        total_frames += 1

        if(display):
          fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg'%(frame))
          im =io.imread(fn)
          ax1.imshow(im)
          plt.title(seq + ' Tracked Targets')

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:
          print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
          if(display):
            d = d.astype(np.int32)
            ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))

        if(display):
          fig.canvas.flush_events()
          plt.draw()
          ax1.cla()

  print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

  if(display):
    print("Note: to get real runtime results run without the option: --display")
```

---

## test/ Folder

### __init__.py
**Purpose:** Empty init file for test module

```python

```

### test_layer1.py
**Purpose:** Test Layer 1 - Frame ingestion

```python
import cv2
from config.layer0_cameras import CAMERAS
from ingest.layer1_frame_ingest import FrameIngestor

def main():
    cam = CAMERAS[0]

    ingestor = FrameIngestor(
        camera_id=cam["camera_id"],
        source=cam["source"],
        sample_rate=3
    )

    for data in ingestor.read():
        frame = data["frame"]
        frame_id = data["frame_id"]
        timestamp = data["timestamp"]

        print(f"[Layer1] Frame {frame_id} | Time {timestamp}")

        cv2.putText(
            frame,
            f"Frame {frame_id}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Layer 1 - Frame Ingest", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    ingestor.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

### test_layer2.py
**Purpose:** Test Layer 2 - YOLO detection

```python
import cv2
from config.layer0_cameras import CAMERAS
from ingest.layer1_frame_ingest import FrameIngestor
from detector.layer2_yolo_detector import YOLODetector

def main():
    cam = CAMERAS[0]

    ingestor = FrameIngestor(
        camera_id=cam["camera_id"],
        source=cam["source"],
        sample_rate=3
    )

    detector = YOLODetector(classes=["person", "mask", "helmet"])

    for data in ingestor.read():
        frame = data["frame"]
        detections = detector.detect(frame)

        print(f"[Layer2] Detections: {len(detections)}")

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class']} {det['confidence']:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imshow("Layer 2 - YOLO Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    ingestor.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

### test_layer3.py
**Purpose:** Test Layer 3 - SORT tracking

```python
import cv2
from config.layer0_cameras import CAMERAS
from ingest.layer1_frame_ingest import FrameIngestor
from detector.layer2_yolo_detector import YOLODetector
from tracker.layer3_sort_tracker import SortTracker

def main():
    cam = CAMERAS[0]

    ingestor = FrameIngestor(
        camera_id=cam["camera_id"],
        source=cam["source"],
        sample_rate=3
    )

    detector = YOLODetector(classes=["person"])
    tracker = SortTracker()

    for data in ingestor.read():
        frame = data["frame"]

        detections = detector.detect(frame)
        tracks = tracker.update(detections)

        for t in tracks:
            x1, y1, x2, y2 = t["bbox"]
            track_id = t["track_id"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        cv2.imshow("Layer 3 - SORT Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    ingestor.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

### test_layer4.py
**Purpose:** Test Layer 4 - Motion analysis

```python
import cv2
from config.layer0_cameras import CAMERAS
from ingest.layer1_frame_ingest import FrameIngestor
from detector.layer2_yolo_detector import YOLODetector
from tracker.layer3_sort_tracker import SortTracker
from tracker.layer4_motion_tracker import MotionAnalyzer

cam = CAMERAS[0]
ingestor = FrameIngestor(cam["camera_id"], cam["source"], sample_rate=3)
detector = YOLODetector(classes=["person"])
tracker = SortTracker()
motion = MotionAnalyzer(frame_gaps=[1,5,10,15,20])

for data in ingestor.read():
    frame = data["frame"]
    frame_id = data["frame_id"]

    detections = detector.detect(frame)
    tracks = tracker.update(detections)
    motion_info = motion.update(tracks, frame_id)

    for m in motion_info:
        print(f"Track {m['track_id']} | Frame {m['frame_id']} | Gaps: {m['motion_gaps']}")
```

---

## Pipeline Architecture

### Layer 0: Configuration
- **File:** `config/layer0_cameras.py`
- **Purpose:** Define camera sources (webcam, video file, RTSP stream)

### Layer 1: Frame Ingestion
- **File:** `ingest/layer1_frame_ingest.py`
- **Purpose:** Capture and sample frames from camera sources
- **Output:** Frame data with timestamp and frame ID

### Layer 2: Object Detection
- **File:** `detector/layer2_yolo_detector.py`
- **Purpose:** Detect objects (person, mask, helmet) using YOLOv8
- **Output:** Bounding boxes with class and confidence

### Layer 3: Object Tracking
- **File:** `tracker/layer3_sort_tracker.py`
- **Purpose:** Track detected objects across frames using SORT algorithm
- **Output:** Tracked objects with unique IDs

### Layer 4: Motion Analysis
- **File:** `tracker/layer4_motion_tracker.py`
- **Purpose:** Analyze motion patterns across multiple frame gaps
- **Output:** Motion metrics for each tracked object

### Layer 5: Behavior Decision
- **File:** `tracker/layer5_behavior.py`
- **Purpose:** Make decisions based on motion patterns (Normal/Warning/Alert)
- **Output:** Behavior classification with reasoning

### Layer 6: Notification
- **File:** `tracker/layer6_telegram.py`
- **Purpose:** Send alerts via Telegram for suspicious behavior
- **Output:** Telegram messages with snapshots

---

## Dependencies

```
opencv-python
ultralytics
filterpy
lap
sort-tracker
scikit-image
requests (for Telegram)
```

---

## Usage

### Run Full Pipeline
```bash
python main.py
```

### Run Individual Layer Tests
```bash
python test/test_layer1.py  # Test frame ingestion
python test/test_layer2.py  # Test YOLO detection
python test/test_layer3.py  # Test SORT tracking
python test/test_layer4.py  # Test motion analysis
```

---

## Key Features

1. **Modular Design:** Each layer is independent and testable
2. **Real-time Processing:** Optimized for live CCTV feeds
3. **Multi-source Support:** Webcam, video files, RTSP streams
4. **Behavior Classification:** Normal, Warning, Alert states
5. **Motion Analysis:** Multi-gap motion tracking for better accuracy
6. **Telegram Integration:** Real-time alerts with snapshots
7. **Visual Feedback:** Color-coded bounding boxes based on behavior

---

## Color Coding

- **Blue:** Normal behavior
- **Orange:** Warning (loitering, mask/helmet detected)
- **Red:** Alert (high motion, fleeing, falling, erratic movement)
