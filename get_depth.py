import time
import numpy as np
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

def depth_callback(msg: Image):
    depth = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
    valid = depth[depth > 0.05]
    if valid.size > 0:
        print(
            f"\rFrame: {msg.width}x{msg.height}  "
            f"min: {valid.min():.3f}m  "
            f"mean: {valid.mean():.3f}m  "
            f"max: {valid.max():.3f}m",
            end="",
            flush=True,
        )

node = Node()
topic = "/depth_camera"

if not node.subscribe(Image, topic, depth_callback):
    raise RuntimeError(f"Failed to subscribe to '{topic}'. Is Gazebo running?")

print(f"Listening to '{topic}' (Ctrl+C to exit)...")
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nStopped.")
