# RoboVerse 2026 Qualifier — Project Context

## Goal
Build a fully autonomous drone mission for the RoboVerse 2026 Qualifier (May 22-23, 2026). University category. Drone searches a randomized 40m x 40m x 8m space port for fuel barrels:
- Yellow barrels (ground level only) = 50 pts each
- Red barrels (elevated only) = 100 pts each
- Time bonus: +20 pts per 30 sec under 5 min for "all of each color"
- 10 minutes total. Manual control = disqualification.
- University eligibility: MUST detect at least one of EACH color or score is unranked.

## Environment
- Ubuntu 22.04 VM (VMware), pre-configured by organizers
- PX4 SITL launched via `~/start_px4.sh`, vehicle option 2 = x500_vision
- Gazebo Harmonic; Python bindings: `gz.transport13`, `gz.msgs10`
- After PX4 start, must set EKF origin:
  `commander set_ekf_origin 47.397742 8.545594 488.0`
- MAVSDK-python is the drone control library
- Project root: `~/Desktop/codes` (this folder), git-initialized

## Key project files (do not delete or rename without confirming)
- `drone_control.py` — MAVSDK wrapper (arm_and_takeoff, send_velocity, send_position_setpoint, rotate_to_yaw, get_yaw, land)
- `depth_receiver.py` — subscribes to /depth_camera, provides get_frame()
- `AvoidancePlanner.py` — depth-to-position planner with clearance + emergency override
- `avoid.py` — working reactive obstacle-avoidance loop
- `avoid_with_detect.py` — MISNAMED, detection not wired in yet (do not assume it works)
- `Detector.py` — threaded YOLO wrapper around ultralytics
- `UseDetectorExample.py` — shows RGB camera subscription pattern
- `get_position_with_task.py` — SharedState + background pose monitor
- `top_down.py` — depth_to_xy_map for projecting depth pixels to camera-frame X-Z
- `save_photo.py` — saves RGB frames (for training data capture)
- `keyboardcontrol.py`, `KEY2.py` — manual flight (data capture only, NOT for the run)
- `Train_YOLO_Models.ipynb` — Ultralytics training notebook for Colab
- `yolov10n.pt` — baseline COCO weights, NOT trained on barrels yet

## Lower-priority files (Finals, not Qualifier)
- `GlobalMapper.py`, `RRTStarPlanner.py`, `PointCloudPlanner.py`, `depthcloud.py`
- `VelocityPlanner.py`, `vel_avoidance.py` (using position-based `avoid.py` instead)

## Key technical facts
- Camera intrinsics: K = [[433, 0, 320], [0, 433, 240], [0, 0, 1]]
- RGB camera topic: `/world/roboverse/model/x500_vision_0/link/camera_link/sensor/IMX214/image`
- Depth topic: `/depth_camera`
- PX4 NED frame: North +X, East +Y, Down +Z, yaw clockwise from north, negative Z = up

## Rules of engagement
- Be concise. No essays. Bullet lists over prose.
- Do not skip ahead in the plan. Confirm current step before starting work.
- Before writing or modifying any file, briefly describe what you'll do and ask me to confirm.
- After any meaningful change, commit to git with a clear message.
- Never run flight or sim commands without asking me first.
- Never delete or overwrite: training data, label files, trained models (.pt/.onnx), or git history.
- If proposing multiple options: recommend one with a one-line tradeoff, don't write essays.
- When debugging: form a hypothesis first, then test it. Don't randomly change code.