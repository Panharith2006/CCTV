import cv2
import time

class FrameIngestor:
    def __init__(self, camera_id, source, sample_rate=3):
        """
        source: int (webcam), str (video file or RTSP)
        """
        self.camera_id = camera_id
        self.source = source
        self.sample_rate = sample_rate
        self.cap = cv2.VideoCapture(source)
        self.frame_count = 0

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open stream/video: {source}")

    def read(self):
        """
        Generator that yields sampled frames
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break  # end of video / camera error

            self.frame_count += 1

            # Sampling: take every N-th frame
            if self.frame_count % self.sample_rate != 0:
                continue

            timestamp = time.time()

            yield {
                "camera_id": self.camera_id,
                "timestamp": timestamp,
                "frame_id": self.frame_count,
                "frame": frame
            }

    def release(self):
        self.cap.release()
