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
- CRITICAL_DISTANCE_M raised from 0.3m to 1.5m per organiser guidance
- FORWARD_SPEED_M_S bumped from 0.8 to 1.2 m/s
- Readiness check uses is_home_position_ok (NOT is_global_position_ok — vision has no GPS)
- Single-pass waypoint tour over 14 calibrated NED waypoints (workshop map)
- Per-waypoint multi-altitude scans (2m/4m/6m) only where needed:
    * box waypoints: all altitudes (yellow + 1-stack + 2-stack + 3-stack red possible)
    * top_room: 2m only (yellow ground level)
    * right_chamber: 4m + 6m (elevated red)
- Goal-directed reactive nav between waypoints (depth avoidance + target heading)
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

    BASE_ALTITUDE_M = 2.0      # cruise altitude between waypoints
    ALL_ALTITUDES_M = [2.0, 4.0, 6.0]
    GROUND_ALT_M = [2.0]       # yellow barrels (ground level only)
    ELEVATED_ALT_M = [4.0, 6.0]  # red barrels (elevated only)

    CLIMB_SPEED_M_S = 1.2
    ALTITUDE_TOLERANCE_M = 0.3
    DETECTION_BURST_FRAMES = 5
    DETECTION_TRIGGER_INTERVAL_S = 2.0
    STUCK_THRESHOLD = 3

    # Single-pass waypoint tour with per-waypoint scan altitudes
    # name, N, E, scan_altitudes (None = transit only)
    WAYPOINTS = [
        ("box_NW",         16.13,  0.38, ALL_ALTITUDES_M),    # SCAN — covers box
        ("box_NE",         15.93, 15.59, ALL_ALTITUDES_M),    # SCAN — covers box other angle
        ("box_exit",       17.59,  8.11, None),               # transit (W back to opening)
        ("corridor",       27.70,  7.77, None),               # transit (N through opening)
        ("tunnel_entry",   29.94, 19.64, None),               # transit (E along upper corridor)
        ("tunnel_bend",    35.62, 18.36, None),               # transit (N up top-room tunnel)
        ("tunnel_aligned", 35.73, 16.15, None),               # transit (W after bend)
        ("top_room",       40.17, 15.84, GROUND_ALT_M),       # SCAN — yellow only
        ("back_aligned",   35.73, 16.15, None),               # backtrack
        ("back_bend",      35.62, 18.36, None),
        ("back_entry",     29.94, 19.64, None),
        ("corridor_east",  27.50, 31.95, None),               # transit (E along corridor)
        ("alcove_entry",   21.54, 32.19, None),               # transit (S into tunnel)
        ("right_chamber",  17.77, 32.43, ELEVATED_ALT_M),     # SCAN — elevated red only
    ]
    WAYPOINT_TOLERANCE_M = 1.5
    WAYPOINT_TIMEOUT_S = 60.0
    HEADING_TOLERANCE_DEG = 20.0
    SCAN_QUARTER_HOLD_S = 1.0
    SCAN_YAW_RATE_DEG_S = 90.0

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
        self.stuck_count = 0

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
            await self.drone.action.set_takeoff_altitude(self.BASE_ALTITUDE_M)
        except Exception:
            pass
        print(f"Takeoff to {self.BASE_ALTITUDE_M:.1f}m")
        await self.drone.action.takeoff()
        async for pos in self.drone.telemetry.position():
            if float(pos.relative_altitude_m) >= self.BASE_ALTITUDE_M - 0.20:
                break
        print("Takeoff complete")
        await asyncio.sleep(2.0)

    async def _poll_position(self):
        async for ned in self.drone.telemetry.position_velocity_ned():
            if not self.running:
                return
            self.current_ned = (ned.position.north_m, ned.position.east_m)

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

    @staticmethod
    def _yaw_error_deg(target, current):
        e = target - current
        while e > 180:
            e -= 360
        while e < -180:
            e += 360
        return e

    def decide_motion_toward(self, left, center, right, target_heading_deg):
        """Goal-directed reactive nav. Heads toward target while respecting depth safety."""
        if self.current_yaw_deg is None:
            return "STOP", self.DECISION_PERIOD_S

        err = self._yaw_error_deg(target_heading_deg, self.current_yaw_deg)

        # All sides blocked → stuck escape
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

        # Center clear → either fly forward (if facing target) or turn toward target
        if center >= self.SAFE_DISTANCE_M:
            if abs(err) < self.HEADING_TOLERANCE_DEG:
                return "FORWARD", self.DECISION_PERIOD_S
            # Turn toward target — duration scales with error
            turn_time = min(abs(err), 45) / self.YAW_RATE_DEG_S
            return ("TURN_RIGHT" if err > 0 else "TURN_LEFT"), turn_time

        # Obstacle ahead — prefer side toward target, but only if that side is clear enough
        prefer_right = err > 0
        if prefer_right and right >= self.CRITICAL_DISTANCE_M:
            return "TURN_RIGHT", self.TURN_DEGREES / self.YAW_RATE_DEG_S
        if (not prefer_right) and left >= self.CRITICAL_DISTANCE_M:
            return "TURN_LEFT", self.TURN_DEGREES / self.YAW_RATE_DEG_S
        # Target side blocked — fall back to clearer side
        if left >= right:
            return "TURN_LEFT", self.TURN_DEGREES / self.YAW_RATE_DEG_S
        return "TURN_RIGHT", self.TURN_DEGREES / self.YAW_RATE_DEG_S

    async def fly_to_waypoint(self, name, target_n, target_e, altitude):
        """Goal-directed nav to (target_n, target_e) at given altitude."""
        print(f"  → {name} ({target_n:.1f}, {target_e:.1f}) @ {altitude:.1f}m")
        loop = asyncio.get_running_loop()
        start = loop.time()
        last_log = 0.0
        while self.running and loop.time() - start < self.WAYPOINT_TIMEOUT_S:
            if self.current_ned is None:
                await asyncio.sleep(0.1)
                continue
            cn, ce = self.current_ned
            dn = target_n - cn
            de = target_e - ce
            dist = math.sqrt(dn * dn + de * de)
            if dist < self.WAYPOINT_TOLERANCE_M:
                print(f"    arrived ({dist:.1f}m from target)")
                return True

            now = loop.time()
            if now - last_log >= 5.0:
                print(f"    dist={dist:.1f}m yaw={self.current_yaw_deg:.0f}°")
                last_log = now

            target_heading_deg = math.degrees(math.atan2(de, dn))

            depth_frame = self.receiver.get_frame()
            if depth_frame is None:
                await self.hold_position(self.DECISION_PERIOD_S)
                continue
            left, center, right = self.compute_clearances(depth_frame)
            action, action_duration = self.decide_motion_toward(left, center, right, target_heading_deg)

            if action == "FORWARD":
                await self.fly_forward(action_duration)
            elif action == "TURN_LEFT":
                await self.yaw_in_place(-self.YAW_RATE_DEG_S, action_duration)
            elif action in ("TURN_RIGHT", "TURN_180"):
                await self.yaw_in_place(self.YAW_RATE_DEG_S, action_duration)
            else:
                await self.hold_position(action_duration)
        print(f"    TIMEOUT reaching {name}")
        return False

    async def scan_360(self):
        """4 quarter turns in place at faster yaw rate, pausing for detection."""
        print("  ↺ scan 360°")
        for _ in range(4):
            await self.yaw_in_place(self.SCAN_YAW_RATE_DEG_S, 90.0 / self.SCAN_YAW_RATE_DEG_S)
            await self.hold_position(self.SCAN_QUARTER_HOLD_S)

    async def scan_multi_altitude(self, altitudes):
        """Scan 360° at each altitude in list. Returns to BASE_ALTITUDE_M after."""
        for alt in altitudes:
            await self._climb_to_altitude(alt)
            await self.scan_360()
        # Restore base altitude for next transit
        if altitudes[-1] != self.BASE_ALTITUDE_M:
            await self._climb_to_altitude(self.BASE_ALTITUDE_M)

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

            # Single-pass waypoint tour with per-waypoint multi-altitude scans
            await self._climb_to_altitude(self.BASE_ALTITUDE_M)
            for name, n, e, scan_alts in self.WAYPOINTS:
                if not self.running:
                    break
                self.stuck_count = 0
                if not await self.fly_to_waypoint(name, n, e, self.BASE_ALTITUDE_M):
                    continue  # skip scan if didn't arrive
                if scan_alts:
                    await self.scan_multi_altitude(scan_alts)

            print("\nMission tour complete — landing")
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
