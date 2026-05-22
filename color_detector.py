import time
import numpy as np
import cv2
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

# OpenCV HSV ranges: H in [0,180], S/V in [0,255]
# Yellow floors kept loose to catch weathered/dirty barrel surfaces (matches red treatment).
YELLOW_LO = np.array((15, 60,  60),  dtype=np.uint8)
YELLOW_HI = np.array((40, 255, 255), dtype=np.uint8)

# Red wraps around H=0, so two bands are needed.
# S/V floors kept loose (70/50) to catch weathered/rusty red surfaces.
RED_LO_1 = np.array((0,   70,  50),  dtype=np.uint8)
RED_HI_1 = np.array((10,  255, 255), dtype=np.uint8)
RED_LO_2 = np.array((160, 70,  50),  dtype=np.uint8)
RED_HI_2 = np.array((180, 255, 255), dtype=np.uint8)

# Barrels are taller than wide; reject square-ish contours (warning stickers etc.)
MIN_ASPECT_RATIO = 1.3


class ColorDetector:
    def __init__(self, min_area: int = 200, open_kernel: int = 5, close_kernel: int = 25,
                 min_aspect_ratio: float = MIN_ASPECT_RATIO):
        self.min_area = min_area
        self.min_aspect_ratio = min_aspect_ratio
        # Small kernel for OPEN (denoise), larger for CLOSE (merge weathering bands)
        self.k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))

    def _mask(self, hsv: np.ndarray, color: str) -> np.ndarray:
        if color == "yellow":
            m = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
        else:  # red
            m = cv2.inRange(hsv, RED_LO_1, RED_HI_1) | cv2.inRange(hsv, RED_LO_2, RED_HI_2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  self.k_open)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.k_close)
        return m

    def _extract(self, mask: np.ndarray, color: str) -> list[dict]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            area = int(cv2.contourArea(c))
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w == 0 or h / w < self.min_aspect_ratio:
                continue
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"]) if M["m00"] else x + w // 2
            cy = int(M["m01"] / M["m00"]) if M["m00"] else y + h // 2
            out.append({
                "bbox": [x, y, x + w, y + h],
                "color": color,
                "area": area,
                "centroid": (cx, cy),
            })
        return out

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        return self._extract(self._mask(hsv, "yellow"), "yellow") + \
               self._extract(self._mask(hsv, "red"),    "red")

    @staticmethod
    def classify_color(crop_bgr: np.ndarray, min_ratio: float = 0.03):
        """Return 'yellow', 'red', or None for a BGR crop based on dominant target color."""
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
        red_mask = cv2.inRange(hsv, RED_LO_1, RED_HI_1) | cv2.inRange(hsv, RED_LO_2, RED_HI_2)
        yellow_count = int(np.count_nonzero(yellow_mask))
        red_count = int(np.count_nonzero(red_mask))
        total = crop_bgr.shape[0] * crop_bgr.shape[1]
        if total == 0:
            return None
        if yellow_count / total < min_ratio and red_count / total < min_ratio:
            return None  # neither colour dominates → reject (likely wall/architecture)
        return "yellow" if yellow_count > red_count else "red"


    @staticmethod
    def annotate(frame_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
        out = frame_bgr.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            colour = (0, 255, 255) if d["color"] == "yellow" else (0, 0, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
            cv2.circle(out, d["centroid"], 4, colour, -1)
            label = f'{d["color"]} a={d["area"]}'
            cv2.putText(out, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        return out


def _live_demo():
    detector = ColorDetector()
    state = {"frames": 0, "last": time.monotonic()}

    def on_image(msg: Image):
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        dets = detector.detect(bgr)
        cv2.imshow("ColorDetector", detector.annotate(bgr, dets))
        cv2.waitKey(1)

        state["frames"] += 1
        now = time.monotonic()
        if now - state["last"] >= 1.0:
            fps = state["frames"] / (now - state["last"])
            print(f"\r{msg.width}x{msg.height}  {fps:5.1f} fps  dets/frame: {len(dets)}", end="", flush=True)
            state["frames"] = 0
            state["last"] = now

    node = Node()
    topic = "/world/roboverse/model/x500_depth_0/link/camera_link/sensor/IMX214/image"
    if not node.subscribe(Image, topic, on_image):
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
    _live_demo()
