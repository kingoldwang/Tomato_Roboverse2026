import time
import numpy as np
import cv2
import os
import torch
from ultralytics import YOLO
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from color_detector import ColorDetector
import asyncio
import queue

TOPIC = "/world/roboverse/model/x500_depth_0/link/camera_link/sensor/IMX214/image"

class GZPhotoDetectorSaver:
    def __init__(self, topic, save_dir="output", model_path="yolov8n.pt", burst_size=30, threshold=0.5):
        self.topic = topic
        self.save_dir = save_dir
        self.burst_size = burst_size
        self.threshold = threshold

        self.img_queue = queue.LifoQueue(maxsize=50)

        if os.path.exists(model_path):
            print(f"Loading model: {model_path} (Threshold: {self.threshold})")
            self.model = YOLO(model_path)
            torch.set_num_threads(2)
            print("Warming up model...")
            self.model(np.zeros((640, 640, 3), dtype=np.uint8), imgsz=640, verbose=False)
            print("Model ready.")
        else:
            print(f"WARNING: Model file '{model_path}' not found. Detection disabled.")
            self.model = None

        self.is_detecting = False
        self.is_saving = False
        self.frames_remaining = 0
        self.display_queue = queue.Queue(maxsize=2)
        self.live_queue = queue.Queue(maxsize=2)

        os.makedirs(self.save_dir, exist_ok=True)

    def trigger_detection_burst(self, numofframes=30):
        if self.model:
            self.burst_size = numofframes
            self.frames_remaining = numofframes
            self.is_detecting = True
            self.is_saving = False
            print("Triggered Camera Detection Task")

    def trigger_capture_burst(self, numofframes=30):
        self.burst_size = numofframes
        self.frames_remaining = numofframes
        self.is_detecting = False
        self.is_saving = True
        print("Triggered Camera Capture Task")

    def _image_callback(self, msg: Image):
        try:
            self.img_queue.put_nowait(msg)
        except queue.Full:
            self.img_queue.queue.clear()
            self.img_queue.put_nowait(msg)

    async def _worker(self):
        print("Camera background worker started.")
        while True:
            try:
                img = self.img_queue.get_nowait()
                await self.loop.run_in_executor(None, self._process_task, img)
                self.img_queue.task_done()
            except queue.Empty:
                await asyncio.sleep(0.01)

    def _process_task(self, img):
        frame_rgb = np.frombuffer(img.data, dtype=np.uint8).reshape((img.height, img.width, 3))
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        try:
            self.live_queue.put_nowait(frame_bgr)
        except queue.Full:
            pass

        if self.frames_remaining > 0 and (self.is_detecting or self.is_saving):
            if self.is_detecting and self.model:
                results = self.model(frame_rgb, conf=self.threshold, imgsz=640, verbose=False)
                annotated = frame_bgr.copy()
                kept_any = False
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    h, w = frame_bgr.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    crop = frame_bgr[y1:y2, x1:x2]
                    hsv_color = ColorDetector.classify_color(crop)
                    if hsv_color is None:
                        continue  # reject: no target colour → likely architecture
                    kept_any = True
                    box_colour = (0, 255, 255) if hsv_color == "yellow" else (0, 0, 255)
                    label = f"{hsv_color}_barrel {float(box.conf[0]):.2f}"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_colour, 2)
                    cv2.putText(annotated, label, (x1, max(0, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_colour, 1, cv2.LINE_AA)
                if kept_any:
                    path = os.path.join(self.save_dir, f"det_{int(time.time() * 1000)}.jpg")
                    cv2.imwrite(path, annotated)
                    try:
                        self.display_queue.put_nowait(annotated)
                    except queue.Full:
                        pass

            elif self.is_saving:
                path = os.path.join(self.save_dir, f"raw_{int(time.time() * 1000)}.jpg")
                cv2.imwrite(path, frame_bgr)

            self.frames_remaining -= 1
            if self.frames_remaining == 0:
                self.is_saving = False
                self.is_detecting = False
                print("Camera task complete.")

    async def _display_loop(self):
        while True:
            try:
                frame = self.live_queue.get_nowait()
                cv2.imshow("Gazebo Live Feed", frame)
                cv2.waitKey(1)
            except queue.Empty:
                pass
            try:
                det = self.display_queue.get_nowait()
                cv2.imshow("Gazebo Photo Booth", det)
                cv2.waitKey(1)
            except queue.Empty:
                pass
            await asyncio.sleep(0.033)

    async def run(self):
        self.loop = asyncio.get_running_loop()
        self.node = Node()
        if self.node.subscribe(Image, self.topic, self._image_callback):
            print(f"Subscribed to {self.topic}")
            asyncio.create_task(self._worker())
            asyncio.create_task(self._display_loop())
            await asyncio.Future()
        else:
            print(f"Failed to subscribe to {self.topic}. Is Gazebo running?")


async def main():
    detector = GZPhotoDetectorSaver(
        topic=TOPIC,
        save_dir="output",
        model_path="barrels_v1.pt",
        burst_size=30,
        threshold=0.5,
    )

    async def auto_trigger():
        await asyncio.sleep(3)  # wait for subscription to settle
        while True:
            detector.trigger_detection_burst(5)
            await asyncio.sleep(2)

    asyncio.create_task(auto_trigger())
    await detector.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...")
