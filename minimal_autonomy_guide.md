# Minimal Autonomy Guide

## Purpose

`minimal_autonomy.py` is a starter autonomy script for a drone with a forward-looking depth camera.

Its purpose is to help teams get a drone flying, moving, and reacting to obstacles quickly, so they can spend more time developing mission ideas such as:

- search
- mapping
- perception
- target-related behaviors

This script is meant to provide a working baseline that teams can understand, run, and improve.

## What The Code Does

When you run `minimal_autonomy.py`, it:

1. Connects to PX4 SITL through MAVSDK.
2. Waits until the drone is ready.
3. Arms and takes off.
4. Starts offboard control.
5. Continuously reads the depth camera.
6. Looks at `left`, `center`, and `right` parts of the scene.
7. Chooses one of four actions:
   - `FORWARD`
   - `TURN_LEFT`
   - `TURN_RIGHT`
   - `STOP`
8. Keeps running until you stop it.
9. Lands safely and disarms when interrupted.

## How The Navigation Logic Works

The decision logic is intentionally simple and easy to follow:

- If the center looks clear, the drone moves forward.
- If the front looks blocked, the drone compares the left and right sides.
- It turns 90 degrees toward the clearer side.
- If everything looks too close, it hovers and keeps checking.

This makes the code a good foundation for teams who want to build more advanced behaviors on top of a working flight loop.

## Files You Need

The main files needed for this baseline are:

- `minimal_autonomy.py`
- `depth_receiver.py`

## How Teams Can Use It

This code gives your team a working autonomy loop. From there, you can build your own ideas on top of it.

You can use it to:

- get the drone flying and reacting to obstacles quickly
- test computer vision modules while the drone is already moving autonomously
- add search behavior on top of a working motion controller
- experiment with mapping or navigation ideas without rebuilding the full flight lifecycle

A useful way to think about it is:

- this script handles the basic flight loop
- your team can focus on making the decision logic smarter

## Where To Add Your Own Logic

The best place to start is the decision-making part of the script.

In particular, teams should look at:

- how depth values are grouped into `left`, `center`, and `right`
- how those values are used to choose an action
- how that action is translated into movement commands

Good places to improve the behavior include:

- changing when the drone chooses `FORWARD`
- changing when it chooses `STOP`
- replacing the simple left/right turn rule with a smarter strategy
- adding wall following
- adding search behavior
- using object detection or mapping outputs to influence movement

## Suggested Development Approach

A good workflow is:

1. Run the baseline successfully first.
2. Confirm the drone can take off, move, turn, and land correctly.
3. Watch the printed outputs:
   - `FORWARD`
   - `TURN_LEFT`
   - `TURN_RIGHT`
   - `STOP`
4. Understand why the script is choosing those actions.
5. Improve one part at a time instead of changing everything at once.

## What This Code Is Best For

This baseline is useful for:

- basic obstacle-aware movement
- simple exploration
- testing perception pipelines during flight
- building higher-level mission logic such as:
  - search patterns
  - mapping
  - object detection response
  - target-driven behavior

## Important Mindset

This is a foundation, not a finished solution. That is useful, because it gives your team something solid to start from while still leaving plenty of room for your own ideas and improvements.

In many cases, the best approach is:

1. run the baseline
2. understand how it makes decisions
3. improve one layer at a time

## How To Run It

1. Start PX4/Gazebo with the depth-camera model.
2. Make sure the depth topic matches what the script expects.
3. Run:

```bash
python3 minimal_autonomy.py
```

## Final Advice

Try to keep the safe flight structure intact:

- take off
- run the task loop
- land safely

That allows your team to focus on improving the intelligence layer of the system while keeping the flight lifecycle clear and reliable.
