import time
import numpy as np
import cv2
import os
from ultralytics import YOLO
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
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
        else:
            print(f"WARNING: Model file '{model_path}' not found. Detection disabled.")
            self.model = None

        self.is_detecting = False
        self.is_saving = False
        self.frames_remaining = 0
        self.show = False

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
        frame = np.frombuffer(img.data, dtype=np.uint8).reshape((img.height, img.width, 3))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        displayframe = None

        if self.frames_remaining > 0 and (self.is_detecting or self.is_saving):
            if self.is_detecting and self.model:
                results = self.model(frame_bgr, conf=self.threshold, verbose=False)
                if len(results[0].boxes) > 0:
                    displayframe = results[0].plot()
                    path = os.path.join(self.save_dir, f"det_{int(time.time() * 1000)}.jpg")
                    cv2.imwrite(path, displayframe)
                    self.show = True

            elif self.is_saving:
                path = os.path.join(self.save_dir, f"raw_{int(time.time() * 1000)}.jpg")
                cv2.imwrite(path, frame_bgr)

            self.frames_remaining -= 1
            if self.frames_remaining == 0:
                self.is_saving = False
                self.is_detecting = False
                print("Camera task complete.")

        if self.show and displayframe is not None:
            cv2.imshow("Gazebo Photo Booth", displayframe)
            cv2.waitKey(1)
            self.show = False

    async def run(self):
        self.loop = asyncio.get_running_loop()
        self.node = Node()
        if self.node.subscribe(Image, self.topic, self._image_callback):
            print(f"Subscribed to {self.topic}")
            asyncio.create_task(self._worker())
            await asyncio.Future()
        else:
            print(f"Failed to subscribe to {self.topic}. Is Gazebo running?")


async def main():
    detector = GZPhotoDetectorSaver(
        topic=TOPIC,
        save_dir="output",
        model_path="yolov8n.pt",
        burst_size=30,
        threshold=0.5,
    )
    await detector.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...")
