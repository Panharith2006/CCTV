# CCTV_AI — Project README

This file documents the CCTV_AI processing pipeline in detail: architecture, how to run, dataset and training instructions, labelling, debugging notes, challenges faced, and recommended next improvements.

---

## 1. Overview

CCTV_AI is a lightweight Python-based pipeline to ingest video frames, detect people, classify head-level attributes (mask / helmet), track objects over time, apply behavioral rules (loitering, erratic movement), and optionally notify via Telegram. The system uses a two-stage detection strategy: a COCO person detector for robust person bounding boxes, and a fine-tuned attribute detector (mask / helmet) run on head crops with a fallback to the full-person crop.

Core components are in these folders:
- `ingest/` — frame ingestion and sampling logic.
- `detector/` — person + attribute detectors (two-stage and single-model wrappers).
- `tracker/` — SORT-based tracker, motion analysis, behavior decision, Telegram notifier.
- `dataset/` — local dataset used for fine-tuning attribute model.
- `runs/` — training/inference outputs (model checkpoints, plots).
- `main.py` — pipeline entry point orchestrating ingest → detect → track → behavior → notify.

---

## 2. Repository Structure (important files)

- `main.py` — orchestrates the live pipeline and debug logging. Run this to start the camera/pipeline.
- `detector/two_stage_detector.py` — COCO person detector + attribute model head-crop logic.
- `detector/layer2_yolo_detector.py` — single-model detector wrapper (used as fallback/diagnostics).
- `tracker/layer3_sort_tracker.py` — SORT tracker wrapper returning track IDs and bboxes.
- `tracker/layer4_motion_tracker.py` — motion smoothing, speed/loiter calculations.
- `tracker/layer5_behavior.py` — rule-based decision engine (mask/helmet => abnormal alert rule present).
- `tracker/layer6_telegram.py` — snapshot + caption Telegram notifier.
- `train_mask.py` — training wrapper for fine-tuning mask/helmet detector (Ultralytics YOLOv8).
- `dataset/mask_dataset/` — dataset root (images/labels split into train/val).
- `collect_training_data.py` — interactive webcam capture helper to collect images for labeling.

---

## 3. Setup & Dependencies

Install dependencies (use a venv). Example with pip:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If `requirements.txt` is not present, install the main libs used:

```bash
pip install opencv-python ultralytics filterpy numpy requests
```

Notes for low-RAM / CPU-only machines (8GB): use `--device cpu` for training and reduce `batch` and `imgsz`. Training will be slower.

---

## 4. How to Run the Pipeline (quickstart)

1. Confirm models exist under `runs/detect/.../weights/best.pt`. If not, run training (see below).
2. Start the pipeline:

```bash
python main.py
```

Main runtime options are set inside `main.py` (model paths, thresholds, sampling rate). The pipeline prints debug info including per-frame counts, track histories, and alert events. For low CPU load, reduce `sample_rate` or `imgsz` inside ingest and detector configs.

---

## 5. Data Collection & Labeling

Goal: collect and label enough head-crop images containing mask/no-mask and helmet/no-helmet examples.

Recommended process:
1. Run the interactive capture tool to gather images from webcam or video sources:

```bash
python collect_training_data.py --out dataset/mask_dataset/images/train --count 500
```

2. Label images in YOLO format. Each image must have a `.txt` file with lines: `class x_center y_center width height` (normalized 0..1). Class mapping used in this project is:
   - 0: person
   - 1: mask
   - 2: helmet

3. Put labels into `dataset/mask_dataset/labels/train` and `labels/val` to match images.

4. Update `data/mask_dataset.yaml` to point to the correct `train`/`val` image folders and class names.

Labeling tools recommended: LabelImg, Roboflow, CVAT, or makesense.ai. Export to YOLO format.

Dataset size guidance: start with at least 500–2,000 labeled images per attribute class (mask / no-mask / helmet / no-helmet) to get usable results. More domain-specific images (same camera angles, lighting) produce better performance.

---

## 6. Training (fine-tuning attribute detector)

Use `train_mask.py` to fine-tune a YOLOv8 model on the `dataset/mask_dataset`.

Example CPU-friendly command:

```bash
python train_mask.py --data data/mask_dataset.yaml --epochs 30 --imgsz 416 --batch 2 --device cpu
```

If GPU available, omit `--device cpu` and increase `batch` and `imgsz`.

Monitor `runs/detect/` for `best.pt` and `results.csv`. After training, point `main.py` to the new checkpoint path.

---

## 7. Detection Logic Details (what the code does)

- Person detection: a COCO-pretrained YOLO model (e.g., `yolov8n.pt`) runs to detect people and produce stable person bboxes.
- Attribute detection: a fine-tuned YOLO model attempts to detect `mask` and `helmet` inside the head crop region. The detector first computes a head region by shrinking the top portion of the person bbox; if no attribute boxes are found there, the system reruns the attribute model on the full-person crop (fallback).
- Tracking: SORT associates detections across frames, assigning `track_id`. Per-track attribute histories (deque) are maintained for temporal smoothing to reduce flicker/false positives.
- Decision rules: If a track consistently shows `mask` or `helmet` within smoothing window and motion isn't unusual, it is considered an alert/abnormal event by current project rules (the user opted to treat mask/helmet as abnormal). Telegram notifications are rate-limited by a cooldown to avoid flooding.

---

## 8. Troubleshooting & Debugging Notes (collected during development)

Observed issues and fixes applied:

- main.py corruption: earlier copies had duplicated/indented blocks causing IndentationError. The file was cleaned and rewritten.
- Attribute model returns few/no boxes on live frames: root cause is an under-sized training dataset (only a couple images originally). Two-stage approach guaranteed person detection, but attribute recall remained poor. Fix: collect more labeled images, augment data, and retrain.
- False positives on faces: added Haar face-check fallback and increased per-class confidence thresholds to reduce misfires.
- Flicker across frames: implemented temporal smoothing per-track using a deque history and majority-vote logic.
- Small attribute area (mask) is hard to detect at low resolutions: increase `imgsz` at training time, use head-crop magnification when running attribute detector, add multi-scale augmentation.
- Telegram snapshots too frequent: added cooldown and per-track suppression.

Debugging tips:
- Enable debug printing in detectors to see raw box counts and model names.
- Run `debug_infer.py` on a single image to verify the attribute model returns boxes before wiring into `main.py`.
- Inspect `runs/detect/<run>/results.csv` and plots for per-class performance.

---

## 9. Challenges Experienced

1. Dataset scarcity — very few labeled attribute examples lead to poor attribute detection and high false negatives/positives.
2. Small object detection — masks occupy few pixels, are occluded, or viewed at odd angles; requires high-resolution crops and focused augmentation.
3. Domain gap — models trained on web images may not generalize to CCTV camera angles and lighting.
4. Real-time constraints — CPU-only inference and tracking require model/size trade-offs to maintain acceptable FPS.
5. Association errors — incorrect track ID assignment in crowded scenes can cause incorrect behavior aggregation.
6. Edge-case camera motions, reflections, and partial occlusions causing inconsistent head crops.

---

## 10. Recommended Next Improvements (prioritized)

1. Data Collection (High priority)
   - Collect at least 2,000 labeled head/face/upper-body images across varied lighting/camera angles, including both mask/no-mask and helmet/no-helmet.
   - Use augmentation (brightness, blur, occlusion, synthetic masks) to increase robustness.

2. Improve Label Quality
   - Ensure correct bounding boxes on head and mask/helmet regions. Consistent labeling helps model converge faster.

3. Model & Training Enhancements
   - Use a larger backbone if hardware allows (yolov8s/yolov8m) to improve small-object recall.
   - Train with higher `imgsz` (512/640), and set multi-scale augmentation.
   - Use focal loss or re-weighting to handle class imbalance.

4. Head/Face Localization
   - Add a dedicated head/face detector (e.g., RetinaFace or a small head detector) to produce more precise head crops for the attribute model.

5. Post-processing & Ensembling
   - Combine attribute model outputs with a lightweight classifier on cropped face/head images (two-step verify) to reduce false positives.
   - Add temporal confidence accumulation and early suppression for unstable tracks.

6. Monitoring & CI
   - Add unit tests for detectors and a small verification dataset with known expected outputs.
   - Automate model evaluation and logging to catch regressions.

7. Efficiency & Deployment
   - Export models to ONNX and use ONNXRuntime or TensorRT for faster inference (if GPU available).
   - Add configuration flags and a small CLI to toggle debug levels without editing code.

8. Privacy & Safety
   - Add optional face-blurring before saving or sending snapshots to protect privacy.

---

## 11. Quick Commands & Examples

Run pipeline:

```bash
python main.py
```

Train model (CPU example):

```bash
python train_mask.py --data data/mask_dataset.yaml --epochs 30 --imgsz 416 --batch 2 --device cpu
```

Collect images (example):

```bash
python collect_training_data.py --out dataset/mask_dataset/images/train --count 500
```

Debug single-image inference:

```bash
python debug_infer.py --img path/to/sample.jpg --weights runs/detect/mask_finetune4/weights/best.pt
```

---

## 12. Action Items / Short-Term Plan

- (Immediate) Label the 942 images copied to `dataset/mask_dataset/images/train`; place YOLO `.txt` files into `dataset/mask_dataset/labels/train`.
- (Next) Collect at least 1,000 additional domain-specific images (same camera, lighting) using `collect_training_data.py`.
- (Then) Retrain `train_mask.py` with the expanded dataset and test on `debug_infer.py`.
- (Follow-up) Add head/face detector integration and refine smoothing thresholds.

---

## 13. Notes & Contacts

If you want, I can:
- Start an interactive labeling helper flow (open images and produce YOLO `.txt` stubs),
- Run the training locally with CPU settings to confirm the pipeline end-to-end,
- Add unit tests for detector wrappers and tracker association.

File created: `Read.md` (this document). Refer to `main.py` for runtime defaults and to `train_mask.py` for training flags.

---

Document created on: 2026-01-07

---

## Appendix — Expanded Layer-by-Layer Reference

The following expands each layer described earlier. Each section contains: Purpose, How it works, a minimal "how to start" code snippet, inputs/outputs, tuning tips, common pitfalls, and quick validation checks.

🔹 LAYER 1 — Camera Input (RTSP)

- Purpose: Obtain raw video frames from a live CCTV camera or a recorded file. This layer is purely an I/O source and should remain free of business logic.
- How it works: Uses OpenCV's `VideoCapture` or a dedicated RTSP client. Handles reconnects, simple buffering, and optional timestamp attachment.
- How to start (minimal):

```python
import cv2
cap = cv2.VideoCapture(rtsp_url)
ret, frame = cap.read()
if not ret:
      # handle reconnect or backoff
      pass
```
- Input: RTSP URL or file path (e.g., `rtsp://user:pass@ip:554/stream` or `video.mp4`).
- Output: Continuously yielded frames (BGR numpy arrays) and timestamps.
- Tuning tips:
   - Use hardware-accelerated decoders where available.
   - Add an exponential backoff for reconnects to avoid log flooding.
   - Attach monotonic timestamp (time.time()) to each frame to avoid clock drift issues.
- Common pitfalls:
   - Dropped frames under high encoding bitrate—reduce resolution at the camera.
   - Blocking reads: use a producer thread and a small queue to decouple decoding from processing.
- Quick validation: open a short file and assert you can read >30 frames in 10 seconds on your system.

🔹 LAYER 2 — Frame Ingest & Sampling

- Purpose: Control how many frames are forwarded to the AI pipeline (rate-limiting) to match compute budget.
- How it works: Either skip frames (process every Nth frame) or use time-based sampling (process frames at target FPS). Optionally resize frames here to a working `imgsz`.
- How to start (example):

```python
sample_every = 3  # process 1 frame out of 3
frame_index = 0
while True:
      ret, frame = cap.read()
      frame_index += 1
      if not ret:
            break
      if frame_index % sample_every != 0:
            continue
      # optionally resize
      frame = cv2.resize(frame, (640, 360))
      yield frame
```
- Input: Raw frames from Layer 1.
- Output: Sampled frames at reduced FPS suitable for model inference.
- Tuning tips:
   - For real-time alerts reduce sample rate; for accuracy increase sample rate.
   - Resize to the `imgsz` used at training if possible to preserve model performance.
- Common pitfalls:
   - Aggressive downsampling can miss short events (e.g., quick gestures).
   - Mixing time-based sampling with frame-count sampling can create irregular windows for trackers.
- Quick validation: run for 1 minute, count yielded frames, and confirm approximate target FPS.

🔹 LAYER 3 — Object Detection (YOLO)

- Purpose: Detect objects of interest (person, mask, helmet) and produce bounding boxes, class labels, and confidences.
- How it works: Loads a YOLOv8 model (pretrained or fine-tuned) and runs inference on each sampled frame or on head crops in a two-stage flow.
- How to start (example using Ultralytics API):

```python
from ultralytics import YOLO
model = YOLO('runs/detect/mask_finetune4/weights/best.pt')
res = model.predict(frame, size=416)
for det in res[0].boxes:
      x1,y1,x2,y2 = map(int, det.xyxy[0])
      cls = int(det.cls[0])
      conf = float(det.conf[0])
```
- Input: Single sampled image (BGR numpy array).
- Output: List of detections [{bbox, class_id, score}].
- Tuning tips:
   - Use `imgsz` consistent with training for best performance.
   - Set per-class confidence thresholds to reduce false positives (e.g., person=0.3, mask=0.5).
   - Consider NMS IoU threshold adjustments if overlapping head/person detections are noisy.
- Common pitfalls:
   - Small-object dropouts: masks are small; increase `imgsz` or magnify head crop.
   - Domain gap: fine-tune on camera-specific images.
- Quick validation: run `debug_infer.py` on a representative frame; verify model returns person + attribute detections as expected.

🔹 LAYER 4 — Tracking (SORT)

- Purpose: Assign persistent IDs to detected people across frames so behavior can be analyzed over time.
- How it works: SORT uses a Kalman filter + Hungarian assignment on detections (usually person bboxes). It predicts tracks between frames and matches new detections.
- How to start (minimal):

```python
from sort.sort import Sort
tracker = Sort(max_age=30, min_hits=3)
dets = np.array([[x1,y1,x2,y2,score], ...])
tracks = tracker.update(dets)
# tracks: [[x1,y1,x2,y2,track_id], ...]
```
- Input: Person bounding boxes from Layer 3.
- Output: Track objects with `track_id`, bbox, and optionally velocity.
- Tuning tips:
   - `max_age`: frames a track can miss before deletion — increase for intermittent detection loss.
   - `min_hits`: start reporting tracks after a minimum number of detections to avoid jitter.
- Common pitfalls:
   - Using mixed detection classes (mask boxes) instead of person-only boxes will break association.
   - Rapid camera pans will make tracks unstable; prefer camera stabilization or ignore during pans.
- Quick validation: visualize tracks on a short sequence and ensure consistent labeling for the same person.

🔹 LAYER 5 — Trajectory Storage

- Purpose: Keep recent trajectory history (positions, timestamps, attributes) in memory for each track to compute motion features.
- How it works: Maintain a per-track deque storing the last N (timestamp, center_x, center_y) entries and attribute flags.
- Example structure:

```python
from collections import deque
track_histories = defaultdict(lambda: deque(maxlen=50))
track_histories[track_id].append((timestamp, cx, cy))
```
- Input: `track_id` and current bbox/center.
- Output: In-memory historical series used by Layer 6.
- Tuning tips:
   - Keep history long enough to capture behaviors (2–5 seconds) but not so long that it delays decisioning.
- Common pitfalls:
   - Memory leak when track deletion is not synchronized—ensure you delete old histories when tracks expire.
- Quick validation: assert `len(track_histories[tid])` increases while a person is present and is cleared after track deletion.

🔹 LAYER 6 — Motion Analysis (RULE-BASED)

- Purpose: Compute simple motion features (speed, direction changes, stopped time) from trajectory history.
- How it works: Use consecutive centers to compute distances and angles, produce smoothed speed and stop-time counters.
- Example computations:

```python
def compute_speed(history):
      # simple last two points
      (t1,x1,y1),(t2,x2,y2) = history[-2], history[-1]
      dist = math.hypot(x2-x1, y2-y1)
      dt = t2 - t1
      return dist / max(dt, 1e-6)
```
- Input: Track history deque.
- Output: Numeric motion features: `speed`, `direction_changes`, `stopped_time`.
- Tuning tips:
   - Convert pixel distances to real-world meters if you have homography/calibration for absolute speed thresholds.
   - Smooth speed with a short moving average to reduce jitter.
- Common pitfalls:
   - Using frame index instead of real timestamps leads to inaccurate speed when frames are dropped.
- Quick validation: simulate a stationary track and assert `speed` ≈ 0 and `stopped_time` increases.

🔹 LAYER 7 — Attribute Check (Mask / Helmet)

- Purpose: Associate mask/helmet detections with person tracks so decisions can consider whether a person has a mask/helmet.
- How it works: For each person bbox, compute a head crop region (top 25–40% of bbox). Run the attribute model on that crop. If empty, run fallback on full person crop. Map attribute boxes back to frame coordinates and assign to the nearest `track_id` by IoU or center proximity.
- How to start (conceptual):

```python
head_crop = frame[y:y+int(h*0.35), x:x+w]
attr_res = attr_model.predict(head_crop)
if not attr_res: 
      attr_res = attr_model.predict(frame[y:y+h, x:x+w])
```
- Input: YOLO detections + `track_id` mapping.
- Output: Per-track flags (`mask`: True/False, `helmet`: True/False) plus per-detection confidences.
- Tuning tips:
   - Increase confidence threshold for attributes to reduce false alarms.
   - Expand head crop when detection fails; experiment with 0.3–0.5 head fraction.
- Common pitfalls:
   - Mistaken association in crowded scenes: prefer IoU + center-distance hybrid matching.
- Quick validation: pick several frames, manually check head crops, and verify attribute detection boxes align with masks/helmets.

🔹 LAYER 8 — Decision Engine (Rules)

- Purpose: Convert motion and attribute signals into human-meaningful statuses (Normal, Warning, Alert) using interpretable rules.
- How it works: A simple rule engine evaluates thresholds and combinations, e.g. mask present → Warning; high speed → Alert; stopped_time > T → Warning/Loitering.
- Example rule set (pseudocode):

```text
IF mask OR helmet -> Warning (user-specific policy)
IF speed > 3.0 m/s -> Alert
IF direction_changes > 5 in 3s -> Warning
IF stopped_time > 30s -> Warning (loitering)
```
- Input: Motion features + Attribute flags.
- Output: Decision object {`track_id`, `status`, `reason`}.
- Tuning tips:
   - Tune thresholds on small labeled validation set and consider per-camera calibration.
   - Use majority vote over the last N decisions to prevent single-frame flips.
- Common pitfalls:
   - Overly aggressive rules create false positives; prefer conservative defaults and escalate after confirmation windows.
- Quick validation: use synthetic tracks with known motion to verify rules fire as expected.

🔹 LAYER 9 — Event Generation

- Purpose: Translate decisions into events and ensure rate-limiting, de-duplication, and enrichment (snapshots, metadata).
- How it works: For each decision, check per-track cooldown and emit event objects saved to storage and optionally sent to notifiers.
- Event object example:

```json
{ "event_type": "Warning", "camera_id": "Cam-2F-East", "track_id": 12, "time": "...", "snapshot": "path/to/jpg", "reason": "mask" }
```
- Input: Decision object.
- Output: Event records, snapshot images, telemetry logs.
- Tuning tips:
   - Implement per-track cooldown (e.g., 60s) to avoid spamming.
   - Keep event severity levels and escalate only after repeated confirmations.
- Common pitfalls:
   - Not persisting event IDs or using unstable snapshot filenames causes deduplication issues.
- Quick validation: trigger a known case and ensure exactly one event is created per cooldown window.

🔹 LAYER 10 — Backend & Storage

- Purpose: Persist events, snapshots, and metadata for later review and audits. Implement retention policies and optional indexing for searchability.
- How it works: Save a JSON record per event in a DB (e.g., SQLite/Postgres) and save snapshots on disk or object storage with a retention TTL job.
- What to store: event_id, camera_id, track_id, status, reason, timestamp, snapshot_path, model_version, confidence_scores.
- Tuning tips:
   - Store model version/hash to tie detections to code/model state.
   - Compress snapshots and periodically aggregate logs to reduce disk use.
- Common pitfalls:
   - Unbounded disk growth; implement retention and periodic cleanups.
- Quick validation: query the DB for recent events and verify snapshot files exist and match event records.

🔹 LAYER 11 — Telegram Notification

- Purpose: Deliver human-readable alerts to operators with contextual information and a snapshot.
- How it works: Build a caption with event summary and send multipart/form-data including the image to Telegram Bot API. Respect rate limits and privacy.
- Example snippet:

```python
import requests
files = {'photo': open(snapshot_path,'rb')}
data = {'chat_id': CHAT_ID, 'caption': caption}
requests.post(f'https://api.telegram.org/bot{TOKEN}/sendPhoto', data=data, files=files)
```
- Input: Event record + snapshot path.
- Output: Telegram message in operator chat.
- Tuning tips:
   - Use short captions; include camera_id, reason, time, and track_id.
   - Add quick-action buttons only if you implement a handler.
- Common pitfalls:
   - Sending high-resolution images increases bandwidth and latency — resize before send.
   - Exposing PII — optionally blur faces before sending.
- Quick validation: trigger a test event and confirm message delivered without duplication.

🔁 COMPLETE DATA FLOW (ONE PERSON) — Expanded

1. `Layer 1` reads frames with timestamps. 2. `Layer 2` samples and resizes frames. 3. `Layer 3` (YOLO) finds people and attributes. 4. `Layer 4` (SORT) assigns `track_id`. 5. `Layer 5` stores per-track histories. 6. `Layer 6` computes motion features. 7. `Layer 7` maps attributes to tracks. 8. `Layer 8` applies rules producing a decision. 9. `Layer 9` emits events with cooldowns. 10. `Layer 10` persists events and snapshots. 11. `Layer 11` notifies via Telegram.

Each layer should have unit tests or smoke checks where possible. Start by validating Layers 1–3 in isolation (can the detector find people and attributes on a few labeled frames?), then validate tracking and motion features using a recorded clip.

---

End of appendix.
