import cv2
import time

RTSP_URL = "rtsp://admin:SDB123456@10.99.100.114:554/Streaming/Channels/101"

print("Connecting to camera...")

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

# Reduce latency
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("ERROR: Cannot connect to camera")
    exit()

print("Connected successfully!")
print("Press Q to quit")

fps_start_time = time.time()
fps = 0
frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Failed to receive frame")
        break

    # Resize for smoother display (optional)
    frame = cv2.resize(frame, (1280, 720))

    # FPS Counter
    frame_count += 1

    if (time.time() - fps_start_time) >= 1:
        fps = frame_count
        frame_count = 0
        fps_start_time = time.time()

    cv2.putText(
        frame,
        f"FPS: {fps}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Hikvision RTSP Stream", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()