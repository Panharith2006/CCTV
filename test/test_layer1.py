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
