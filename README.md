# RoboVerse 2026 Qualifier

## Scoring

- Yellow barrel (ground only): **50 pts** each
- Red barrel (elevated only): **100 pts** each
- Time bonus: **+20 pts / 30 sec** under 5 min (all-of-each-color)
- **Must detect at least 1 yellow AND 1 red** or score is unranked
- Manual control = disqualified

## How to run

**Terminal 1 — start sim:**
```bash
~/start_px4.sh
# pick option 2 (x500_depth)
# pick option 1 (roboverse)
# pick option 2 (No QGC)
# wait for pxh> prompt + Gazebo window
```

**Terminal 2 — run a script:**
```bash
cd ~/Desktop/codes
python3 <script>.py
```

**If `start_px4.sh` says "PX4 server already running":**
```bash
ps aux | grep px4    # find stuck PID
kill -9 <PID>        # pkill won't work, processes are stopped
```
---

## What works currently

- Sensors: position, depth, battery, video
- `drone_control.Drone()`: arm / takeoff / yaw / position setpoint / land
- `avoid.py`: reactive obstacle avoidance (60 s bounded)

## What's left

- [ ] Capture training data (~600 frames, fly + `save_photo.py`)
- [ ] Label in LabelImg (`yellow_barrel`, `red_barrel`)
- [ ] Train YOLO on Colab (`Train_YOLO_Models.ipynb`, 100 epochs, target mAP > 0.85)
- [ ] Verify detection live (`UseDetectorExample.py`)
- [ ] Build `mission_main.py` — sweep + detect + dedup + land
---

## File guide

**Core flight:**
- `drone_control.py` — MAVSDK wrapper
- `depth_receiver.py` — depth camera subscriber
- `AvoidancePlanner.py` — depth → avoidance vector
- `avoid.py` — reactive avoidance loop
- `get_position_with_task.py` — pose monitor task

**Diagnostics:**
- `get_position.py`, `get_depth.py`, `get_battery.py`, `get_video.py`
- `basic_offboard.py`, `test_drone_control.py`

**Detection (not yet verified):**
- `Detector.py` — threaded YOLO wrapper
- `UseDetectorExample.py` — RGB subscription pattern
- `Train_YOLO_Models.ipynb` — Colab training
- `yolov10n.pt` — **COCO weights only, not trained on barrels**
- `top_down.py` — depth → camera-frame X-Z projection

**Manual / data capture for testing:**
- `keyboardcontrol.py` — preferred manual flight
- `save_photo.py` — saves RGB frames for training
- `KEY2.py` — older variant, ignore

**Do not touch:**
- `avoid_with_detect.py` — misnamed, detection not wired in
- Finals-only files: `GlobalMapper.py`, `RRTStarPlanner.py`, `PointCloudPlanner.py`, `depthcloud.py`, `VelocityPlanner.py`, `vel_avoidance.py`

---

## Key facts

- Camera K = `[[433, 0, 320], [0, 433, 240], [0, 0, 1]]`
- RGB topic: `/world/roboverse/model/x500_depth_0/link/camera_link/sensor/IMX214/image`
- Depth topic: `/depth_camera`
- PX4 NED: North +X, East +Y, **Down +Z** (negative Z = up), yaw clockwise from north
- Battery auto-pinned to 99% by `drone_control.Drone.connect()`
- EKF origin auto-handled in `x500_depth` mode (no manual command needed)
