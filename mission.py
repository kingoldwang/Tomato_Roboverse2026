#!/usr/bin/env python3
"""
RoboVerse 2026 Qualifier — autonomous mission.

**Qualifier uses x500_vision drone (no GPS).** Pre-flight checklist:
  1. Start sim: ~/start_px4.sh → option 1 (x500_vision) → roboverse → no QGC
  2. In pxh> shell BEFORE running this script:
       commander set_ekf_origin 47.397742 8.545594 488.0
  3. Wait for "Setting GPS origin" then "EKF (xxxxx) home position set"
  4. Then run: python3 mission.py

Without step 2, is_home_position_ok will never be True and arming will fail.

Extends MinimalAutonomy (organiser baseline):
- Reactive depth-based navigation kept intact
- CRITICAL_DISTANCE_M raised from 0.3m to 1.5m per organiser guidance
- FORWARD_SPEED_M_S bumped from 0.8 to 1.2 m/s
- Readiness check uses is_home_position_ok (NOT is_global_position_ok — vision has no GPS)
- Visited-cell tracking biases turn direction toward unexplored areas
- Three altitude phases: 2m (yellow + 1-stack red), 4m (2-stack red), 6m (3-stack red)
- YOLO + HSV detection runs continuously as background asyncio task
- Detection JPGs saved to output/ as evidence for the judge
"""

import asyncio
import math

from mavsdk.action import ActionError

from minimal_autonomy import MinimalAutonomy
from gzphotodetectorsaver import GZPhotoDetectorSaver, TOPIC as RGB_TOPIC


class Mission(MinimalAutonomy):
    CRITICAL_DISTANCE_M = 1.5
    FORWARD_SPEED_M_S = 1.2

    PHASE_DURATIONS_S = [240, 180, 120]
    PHASE_ALTITUDES_M = [2.0, 4.0, 6.0]

    CLIMB_SPEED_M_S = 0.8
    ALTITUDE_TOLERANCE_M = 0.3
    DETECTION_BURST_FRAMES = 5
    DETECTION_TRIGGER_INTERVAL_S = 2.0

    CELL_SIZE_M = 4.0          # arena grid is 4×4m per organiser
    SIMILAR_CLEARANCE_M = 1.0  # L/R within this → use visited info as tiebreak
    STUCK_THRESHOLD = 3        # consecutive STOPs before forcing a 180° escape

    def __init__(self, depth_topic="/depth_camera", rgb_topic=RGB_TOPIC, model_path="barrels_v2.pt"):
        super().__init__(depth_topic=depth_topic)
        self.detector = GZPhotoDetectorSaver(
            topic=rgb_topic,
            save_dir="output",
            model_path=model_path,
            threshold=0.5,
        )
        self.current_ned = None
        self.current_yaw_deg = None
        self.visited_cells = set()
        self.stuck_count = 0

    def _cell_for(self, x, y):
        return (int(x // self.CELL_SIZE_M), int(y // self.CELL_SIZE_M))

    def _predict_turn_cell(self, direction):
        """Cell drone would enter after a 90° turn + forward step."""
        if self.current_ned is None or self.current_yaw_deg is None:
            return None
        x, y = self.current_ned
        delta = -90 if direction == "left" else 90
        new_yaw_rad = math.radians(self.current_yaw_deg + delta)
        # NED: yaw 0 = +X (North), yaw 90 = +Y (East), clockwise
        dx = math.cos(new_yaw_rad) * self.CELL_SIZE_M
        dy = math.sin(new_yaw_rad) * self.CELL_SIZE_M
        return self._cell_for(x + dx, y + dy)

    def decide_motion(self, left, center, right):
        if center >= self.SAFE_DISTANCE_M:
            self.stuck_count = 0
            return "FORWARD", self.DECISION_PERIOD_S

        if (left < self.CRITICAL_DISTANCE_M
                and center < self.CRITICAL_DISTANCE_M
                and right < self.CRITICAL_DISTANCE_M):
            self.stuck_count += 1
            if self.stuck_count >= self.STUCK_THRESHOLD:
                self.stuck_count = 0
                print("  ESCAPE: stuck — turning 180°")
                return "TURN_180", 180.0 / self.YAW_RATE_DEG_S
            return "STOP", self.DECISION_PERIOD_S

        self.stuck_count = 0

        # Tiebreak by visited-cell info when L/R clearances are close
        if abs(left - right) < self.SIMILAR_CLEARANCE_M:
            left_cell = self._predict_turn_cell("left")
            right_cell = self._predict_turn_cell("right")
            if left_cell is not None and right_cell is not None:
                left_unvisited = left_cell not in self.visited_cells
                right_unvisited = right_cell not in self.visited_cells
                if left_unvisited and not right_unvisited:
                    return "TURN_LEFT", self.TURN_DEGREES / self.YAW_RATE_DEG_S
                if right_unvisited and not left_unvisited:
                    return "TURN_RIGHT", self.TURN_DEGREES / self.YAW_RATE_DEG_S

        if left >= right:
            return "TURN_LEFT", self.TURN_DEGREES / self.YAW_RATE_DEG_S
        return "TURN_RIGHT", self.TURN_DEGREES / self.YAW_RATE_DEG_S

    async def wait_until_ready(self, timeout_s=60.0):
        """Vision drone readiness: wait for is_home_position_ok (NOT GPS).

        Requires `commander set_ekf_origin` to have been run in pxh> shell.
        """
        print("Waiting for is_home_position_ok=True (vision drone, no GPS)...")
        print("  → If this hangs, run in pxh>: commander set_ekf_origin 47.397742 8.545594 488.0")
        loop = asyncio.get_running_loop()
        start = loop.time()
        last_status = 0.0
        async for health in self.drone.telemetry.health():
            home_ok = getattr(health, "is_home_position_ok", False)
            local_ok = getattr(health, "is_local_position_ok", False)
            armable = getattr(health, "is_armable", False)
            if home_ok and local_ok and armable:
                print("Ready (home_ok=True, local_ok=True, armable=True)")
                return
            now = loop.time()
            if now - last_status >= 5.0:
                print(f"  ... home={home_ok}, local={local_ok}, armable={armable}")
                last_status = now
            if now - start > timeout_s:
                raise TimeoutError(
                    f"Timeout waiting for readiness "
                    f"(home={home_ok}, local={local_ok}, armable={armable}). "
                    f"Did you run 'commander set_ekf_origin ...' in pxh>?"
                )
            await asyncio.sleep(0.5)

    async def arm_and_takeoff(self):
        """Override base: strict readiness wait + arm retry with delay."""
        await self.wait_until_ready()
        for attempt in range(3):
            try:
                print(f"Arming (attempt {attempt + 1}/3)...")
                await self.drone.action.arm()
                break
            except ActionError as e:
                if attempt == 2:
                    raise RuntimeError(f"Arm failed after 3 attempts: {e}") from e
                print(f"  arm denied, retrying in 3s: {e}")
                await asyncio.sleep(3.0)

        try:
            await self.drone.action.set_takeoff_altitude(2.0)
        except Exception:
            pass
        print("Takeoff to 2.0m")
        await self.drone.action.takeoff()
        async for pos in self.drone.telemetry.position():
            if float(pos.relative_altitude_m) >= 2.0 - 0.20:
                break
        print("Takeoff complete")
        await asyncio.sleep(2.0)

    async def _poll_position(self):
        async for ned in self.drone.telemetry.position_velocity_ned():
            if not self.running:
                return
            self.current_ned = (ned.position.north_m, ned.position.east_m)
            self.visited_cells.add(self._cell_for(*self.current_ned))

    async def _poll_attitude(self):
        async for att in self.drone.telemetry.attitude_euler():
            if not self.running:
                return
            self.current_yaw_deg = att.yaw_deg

    async def _trigger_detection_loop(self):
        await asyncio.sleep(3.0)
        while self.running:
            self.detector.trigger_detection_burst(self.DETECTION_BURST_FRAMES)
            await asyncio.sleep(self.DETECTION_TRIGGER_INTERVAL_S)

    async def _climb_to_altitude(self, target_alt_m):
        print(f"Climbing to {target_alt_m:.1f}m")
        async for pos in self.drone.telemetry.position():
            if not self.running:
                return
            current_alt = float(pos.relative_altitude_m)
            err = target_alt_m - current_alt
            if abs(err) < self.ALTITUDE_TOLERANCE_M:
                break
            down_cmd = -self.CLIMB_SPEED_M_S if err > 0 else self.CLIMB_SPEED_M_S
            await self.set_body_velocity(0.0, 0.0, down_cmd, 0.0)
        await self.set_body_velocity(0.0, 0.0, 0.0, 0.0)
        await self.hold_position(1.0)
        print(f"  reached {target_alt_m:.1f}m")

    async def _run_phase(self, duration_s):
        loop = asyncio.get_running_loop()
        end_time = loop.time() + duration_s
        prev_action = None
        step = 0
        while self.running and loop.time() < end_time:
            depth_frame = self.receiver.get_frame()
            if depth_frame is None:
                await self.hold_position(self.DECISION_PERIOD_S)
                continue
            left, center, right = self.compute_clearances(depth_frame)
            action, action_duration = self.decide_motion(left, center, right)
            # Log on action change, on STOP/ESCAPE, or every 10 steps
            should_log = (action != prev_action) or (action in ("STOP", "TURN_180")) or (step % 10 == 0)
            if should_log:
                remaining = int(end_time - loop.time())
                visited_n = len(self.visited_cells)
                print(f"{action} | L={left:.2f} C={center:.2f} R={right:.2f} | cells={visited_n} | t-{remaining}s")
            prev_action = action
            step += 1
            if action == "FORWARD":
                await self.fly_forward(action_duration)
            elif action == "TURN_LEFT":
                await self.yaw_in_place(-self.YAW_RATE_DEG_S, action_duration)
            elif action == "TURN_RIGHT":
                await self.yaw_in_place(self.YAW_RATE_DEG_S, action_duration)
            elif action == "TURN_180":
                await self.yaw_in_place(self.YAW_RATE_DEG_S, action_duration)
            else:
                await self.hold_position(action_duration)

    async def run(self):
        print("Starting RoboVerse Qualifier mission")
        bg_tasks = []
        try:
            bg_tasks.append(asyncio.create_task(self.detector.run()))
            bg_tasks.append(asyncio.create_task(self._trigger_detection_loop()))

            await self.connect()
            bg_tasks.append(asyncio.create_task(self._poll_position()))
            bg_tasks.append(asyncio.create_task(self._poll_attitude()))

            await self.arm_and_takeoff()
            await self.start_offboard()

            for i, (duration, altitude) in enumerate(zip(self.PHASE_DURATIONS_S, self.PHASE_ALTITUDES_M)):
                if not self.running:
                    break
                print(f"\n=== PHASE {i+1}/{len(self.PHASE_ALTITUDES_M)}: altitude {altitude:.1f}m, duration {duration}s ===")
                self.visited_cells.clear()  # fresh exploration map per altitude
                await self._climb_to_altitude(altitude)
                await self._run_phase(duration)

            print("\nMission phases complete")
        except asyncio.CancelledError:
            print("Mission cancelled")
            raise
        finally:
            self.running = False
            for t in bg_tasks:
                if not t.done():
                    t.cancel()
            await self.shutdown()


async def main():
    mission = Mission()
    try:
        await mission.run()
    except KeyboardInterrupt:
        mission.stop()
        await mission.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
