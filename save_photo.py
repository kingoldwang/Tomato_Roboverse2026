import time
import numpy as np
import cv2
import os
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

SAVE_INTERVAL_S = 1.5
SAVE_DIR = "captured_images"
TOPIC = "/world/roboverse/model/x500_depth_0/link/camera_link/sensor/IMX214/image"

os.makedirs(SAVE_DIR, exist_ok=True)

existing = [f for f in os.listdir(SAVE_DIR) if f.startswith("frame_") and f.endswith(".jpg")]
start_count = max([int(f[6:10]) for f in existing], default=-1) + 1

state = {"count": start_count, "last_save": 0.0}

def image_callback(msg: Image):
    frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow("Gazebo Live Feed", frame_bgr)
    cv2.waitKey(1)

    now = time.monotonic()
    if now - state["last_save"] < SAVE_INTERVAL_S:
        return
    state["last_save"] = now

    filename = os.path.join(SAVE_DIR, f"frame_{state['count']:04d}.jpg")
    cv2.imwrite(filename, frame_bgr)
    state["count"] += 1
    print(f"Saved: {filename}")

def main():
    node = Node()
    if not node.subscribe(Image, TOPIC, image_callback):
        print(f"Failed to subscribe to {TOPIC}. Is Gazebo running?")
        return
    print(f"Subscribed to {TOPIC}")
    print(f"Saving 1 frame every {SAVE_INTERVAL_S}s to '{SAVE_DIR}/'. Ctrl+C to stop.")

    try:
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        print(f"\nStopped. Total saved: {state['count']}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()