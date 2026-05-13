import time
import numpy as np
import cv2
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

frame_count = 0
last_print = time.monotonic()

def image_callback(msg: Image):
    global frame_count, last_print

    frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Gazebo Live Feed", frame_bgr)
    cv2.waitKey(1)

    frame_count += 1
    now = time.monotonic()
    if now - last_print >= 1.0:
        fps = frame_count / (now - last_print)
        print(f"\r{msg.width}x{msg.height}  {fps:5.1f} fps  total: {frame_count}", end="", flush=True)
        frame_count = 0
        last_print = now

def main():
    node = Node()
    topic = "/world/roboverse/model/x500_depth_0/link/camera_link/sensor/IMX214/image"

    if not node.subscribe(Image, topic, image_callback):
        print(f"Failed to subscribe to {topic}. Is Gazebo running?")
        return

    print(f"Subscribed to {topic} (Ctrl+C to stop)...")
    try:
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
