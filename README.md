# CCTV AI Project

A comprehensive, AI-powered CCTV surveillance system built with Python, PyTorch, and OpenCV. This project provides a real-time pipeline for detecting people, verifying Personal Protective Equipment (PPE) like masks and helmets, tracking individuals across frames, and sending real-time alerts via Telegram.

## Features

- **Real-Time Object Detection:** Utilizes a custom YOLO model (`ultralytics`) to detect persons, masks, and helmets in video streams.
- **Robust Tracking:** Implements the SORT (Simple Online and Realtime Tracking) algorithm to track individuals consistently across consecutive frames.
- **Person Re-Identification (ReID):** Integrates `torchreid` to maintain person identity even if they leave and re-enter the camera's field of view.
- **PPE Compliance Monitoring:** Associates detected masks and helmets with specific tracked individuals. Uses OpenCV's Haar cascades for lightweight face detection to improve mask association.
- **Database Integration:** Connects to a MySQL database to log person statistics and store persistent identities.
- **Telegram Alerts:** Automatically sends full-frame and cropped image alerts to a configured Telegram chat when abnormal conditions (e.g., person wearing a mask or helmet) are detected.

## Technology Stack

### Core Technologies
- **Language:** Python 3
- **Deep Learning Framework:** PyTorch
- **Computer Vision:** OpenCV (`opencv-python`)
- **Object Detection:** Ultralytics YOLO (v8/v9 depending on `best.pt` model)
- **Object Tracking:** SORT (`filterpy`, `numpy`, `scipy`)
- **Re-Identification:** `torchreid`

### Utilities & Integrations
- **Database:** MySQL (`mysql-connector-python`)
- **Notifications:** Telegram Bot API (`requests`)
- **Environment Management:** `python-dotenv`
- **File Downloader:** `gdown` (for fetching models/weights if needed)
- **Data Manipulation:** `numpy`, `matplotlib`, `scikit-image`

## Project Structure

```
CCTV_AI/
│
├── config/              # Configuration files (Cameras, Telegram credentials)
├── database/            # Database connection and schema operations
├── detector/            # Object detection modules (YOLO implementations)
├── ingest/              # Frame ingestion from camera sources
├── sort/                # SORT tracking algorithm dependencies/modules
├── test/                # Unit tests and testing scripts
├── thumbnails/          # Directory for storing cropped images for alerts
├── tracker/             # Tracking logic, ReID Manager, and Telegram Notifier
├── venv/                # Python virtual environment
│
├── .env                 # Environment variables (DB credentials, Telegram tokens)
├── best.pt              # Custom trained YOLO model weights
├── main.py              # Main application pipeline and entry point
└── requirements.txt     # Python package dependencies
```

## Setup and Installation

### 1. Prerequisites
- Python 3.8 or higher.
- MySQL Server installed and running.
- A Telegram Bot Token and Chat ID for notifications.

### 2. Clone the Repository
Clone this repository to your local machine:
```bash
git clone <repository-url>
cd CCTV_AI
```

### 3. Create a Virtual Environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Environment Variables
Copy the `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```
Ensure your `.env` contains the necessary variables for:
- Database connection strings
- `BOT_TOKEN`
- `CHAT_ID`

### 6. Model Weights
Ensure that your trained YOLO model (`best.pt`) is placed in the root directory of the project.

## Usage

To start the CCTV pipeline, run the main script:

```bash
python main.py
```

Press `ESC` while the video window is in focus to stop the program safely.

## Technical Architecture & Deep Dive (How It Works)

This project integrates several state-of-the-art computer vision algorithms into a seamless pipeline. Below is a detailed breakdown of how each component functions under the hood:

### 1. Frame Ingestion
The `FrameIngestor` connects to video sources (RTSP streams, webcams, or video files) using OpenCV's `VideoCapture`. To ensure the system runs in real-time, it implements a sampling rate mechanism (e.g., processing every 3rd frame), dropping frames if the processing pipeline falls behind, which prevents a backlog and keeps the system strictly "live".

### 2. Object Detection (YOLO)
**Technology:** Ultralytics YOLO (You Only Look Once)
**How it works:** 
- Instead of using a two-stage approach (like R-CNN), YOLO passes the entire image through a Convolutional Neural Network (CNN) in a single forward pass.
- The CNN divides the image into a grid and simultaneously predicts bounding boxes, confidence scores, and class probabilities (Person, Mask, Helmet) for each grid cell.
- **Non-Maximum Suppression (NMS):** To prevent multiple bounding boxes for the same object, the system applies NMS. It looks at boxes that highly overlap (high Intersection over Union, or IoU) and keeps only the box with the highest confidence score, suppressing the rest.
- **Why YOLO?:** It provides the perfect balance between high accuracy and real-time inference speed necessary for live CCTV analysis.

### 3. Object Tracking (SORT)
**Technology:** Simple Online and Realtime Tracking (SORT)
**How it works:** 
Detection alone doesn't tell us if a person in frame 1 is the same person in frame 2. Tracking solves this.
- **Kalman Filters:** For every detected person, a Kalman Filter predicts where that person will be in the next frame based on their current velocity and position.
- **Hungarian Algorithm:** When the next frame arrives with new YOLO detections, the Hungarian algorithm optimally matches the new detections to the predicted positions from the Kalman Filter by maximizing the overlap (IoU).
- **Result:** Each person is assigned a persistent, unique "Track ID" as long as they remain in the camera's view.

### 4. PPE Attribute Association & Temporal Smoothing
**Technology:** Spatial overlap and Heuristics
**How it works:** 
- Once a person is tracked, the system checks if a detected mask or helmet belongs to them. It does this by calculating if the bounding box of the PPE falls within the upper portion (the head region) of the person's bounding box.
- **Haar Cascades for Faces:** To improve mask detection accuracy, a lightweight OpenCV Haar Cascade detects faces. If a mask overlaps heavily with a detected face, the confidence of the association skyrockets.
- **Temporal Smoothing:** AI detections can flicker (a mask might be missed in 1 out of 10 frames). The system uses a history queue (e.g., tracking the last 30 frames). A person is only officially flagged as "wearing a mask" if the mask is detected for a minimum required threshold (e.g., 3 out of the last 30 frames). This practically eliminates false positives/negatives.

### 5. Person Re-Identification (ReID)
**Technology:** `torchreid` (Deep Metric Learning)
**How it works:** 
SORT tracking fails if a person walks behind a pillar or leaves the camera view and returns. ReID solves this long-term tracking issue.
- When a person is confirmed by SORT to be stable, their image is cropped and passed through a ReID neural network (like OSNet or ResNet).
- The network outputs a high-dimensional vector (an "embedding" or "feature signature") representing the visual appearance of the person's clothing and body.
- When a "new" track appears, the system extracts its embedding and compares it against a database of known embeddings using **Cosine Similarity**. If the similarity score is above a certain threshold (e.g., 0.7), the system recognizes them as a previously seen individual and restores their original Identity ID.

### 6. Database & Alerting System
**Technology:** MySQL & Telegram Bot API
**How it works:** 
- **MySQL Database:** Stores the ReID embeddings, person names, and statistical data to ensure persistence even if the software is restarted.
- **Telegram Alerting:** The system monitors the state of tracked individuals. If a person is assigned an ID and is detected wearing restricted PPE (like a mask or helmet), the system captures the full frame and a zoomed-in crop of the person. It then triggers an asynchronous HTTP POST request to the Telegram Bot API, delivering a real-time photo and text alert directly to a security team's mobile devices without interrupting the main video processing loop.
