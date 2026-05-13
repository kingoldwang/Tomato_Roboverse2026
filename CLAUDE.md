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
- PX4 SITL launched via `~/start_px4.sh`. Vehicle options:
  - Option 1 = `x500_vision` (VIO-based, no GPS — only use if testing vision pipeline)
  - Option 2 = `x500_depth` (depth camera + simulated GPS — default for flying)
- Gazebo Harmonic; Python bindings: `gz.transport13`, `gz.msgs10`
- EKF origin only needs to be set manually in `x500_vision` mode:
  `commander set_ekf_origin 47.397742 8.545594 488.0`
- MAVSDK-python is the drone control library
- Project root: `~/Desktop/codes` (this folder), git-initialized

## Startup sequence
Two terminals:
1. `~/start_px4.sh` → option 2 (x500_depth) → option 1 (roboverse) → option 2 (No QGC). Wait for `pxh>` prompt and Gazebo window.
2. In another terminal: `cd ~/Desktop/codes && python3 <script>.py`

If `start_px4.sh` fails with "PX4 server already running": leftover processes are in `T` (stopped) state. `pkill` won't work — kill by PID with `kill -9`.

EKF origin is auto-handled by GPS in `x500_depth`. Battery is pinned at 99% by scripts that call `drone_control.Drone.connect()`.

## Key project files (do not delete or rename without confirming)
Core flight stack (verified at checkpoint-1):
- `drone_control.py` — MAVSDK wrapper. Pins SIM_BAT_MIN_PCT=99, waits for EKF, polls altitude/landed_state instead of fixed sleeps.
- `depth_receiver.py` — subscribes to /depth_camera, provides get_frame()
- `AvoidancePlanner.py` — depth-to-position planner with clearance + emergency override
- `avoid.py` — reactive obstacle-avoidance loop. First-data wait, 60s runtime cap, lands on exit.
- `get_position_with_task.py` — SharedState + background pose monitor task

Diagnostics (Step 2, all verified):
- `get_position.py` — passive NED pose reader
- `get_depth.py` — depth-camera stats reader
- `get_battery.py` — battery telemetry reader
- `get_video.py` — RGB feed window + FPS counter
- `basic_offboard.py` — minimal offboard velocity smoke test
- `test_drone_control.py` — Step 3 end-to-end Drone() wrapper test

Detection stack (not yet verified — Step 5):
- `Detector.py` — threaded YOLO wrapper around ultralytics
- `UseDetectorExample.py` — RGB subscription pattern reference
- `Train_YOLO_Models.ipynb` — Ultralytics training notebook for Colab
- `yolov10n.pt` — baseline COCO weights, NOT trained on barrels yet
- `top_down.py` — depth_to_xy_map for projecting depth pixels to camera-frame X-Z

Manual control / data capture (not for the qualifier run):
- `keyboardcontrol.py` — rewritten: stateless velocity, cbreak terminal, battery pin
- `KEY2.py` — older variant, prefer keyboardcontrol.py
- `save_photo.py` — saves RGB frames for training data capture

Stale / risky:
- `avoid_with_detect.py` — MISNAMED, detection not wired in yet (do not assume it works)

## Lower-priority files (Finals, not Qualifier)
- `GlobalMapper.py`, `RRTStarPlanner.py`, `PointCloudPlanner.py`, `depthcloud.py`
- `VelocityPlanner.py`, `vel_avoidance.py` (using position-based `avoid.py` instead)

## Checkpoint-1 status (foundation done)
Verified working in SITL with x500_depth + roboverse:
- All four sensor streams (position, depth, battery, video)
- `drone_control.Drone()` arm/takeoff/yaw-rotate/position-setpoint/land
- `avoid.py` reactive obstacle-avoidance loop (60 s bounded flights)

Not yet verified:
- YOLO detection pipeline (`Detector.py` + `UseDetectorExample.py`) — pending
- Barrel-trained model — `yolov10n.pt` is still COCO weights; training data not collected
- Search pattern / full mission integration — not started

## Key technical facts
- Camera intrinsics: K = [[433, 0, 320], [0, 433, 240], [0, 0, 1]]
- RGB camera topic: `/world/roboverse/model/x500_depth_0/link/camera_link/sensor/IMX214/image`
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
- Short-term fixes must not compromise the long-term mission. When patching an error, name the long-term implication and prefer reusable, well-placed fixes over throwaway hacks. If a quick fix is the right call, flag it as temporary and note what should eventually replace it. Never bury sim-only or debug-only logic in core files without clearly marking it.