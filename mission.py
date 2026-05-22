#!/usr/bin/env python3
"""RoboVerse 2026 Qualifier mission — local cost-grid + DWA + frontier.

Architecture:
  - Background tasks: position + attitude pollers, depth/RGB receivers, detector.
  - Control loop (10 Hz):
      1. Recentre cost grid on current drone pose; decay; mark visited.
      2. Integrate latest depth frame into the grid.
      3. Pick a frontier goal if we don't have one (or current is reached/stale).
      4. Run DWA-lite to choose a (forward_speed, yaw_rate) for the next tick.
      5. Command the drone with set_body_velocity.
  - Altitude phases (2m → 4m → 6m) cycle on a timer.
  - Periodic 360° scans every PERIODIC_SCAN_INTERVAL_S to look around.

There are no hardcoded waypoints. The goal is always derived from the local
cost grid — the planner only knows: "this direction is free / unknown / blocked".

Vision-drone pre-flight:
  1. ~/start_px4.sh → option 1 (x500_vision) → roboverse → no QGC
  2. In pxh>: commander set_ekf_origin 47.397742 8.545594 488.0
  3. python3 mission.py
"""

import asyncio
import math
import time

from mavsdk.action import ActionError

from minimal_autonomy import MinimalAutonomy
from gzphotodetectorsaver import GZPhotoDetectorSaver, TOPIC as RGB_TOPIC
from local_planner import (
    CostGrid, GridConfig,
    DwaPlanner, DwaConfig,
    FrontierPicker, FrontierConfig,
)


class Mission(MinimalAutonomy):
    # Altitude phases — heights to revisit, not waypoints.
    # Strategy: prioritise 2.3m (ground yellows = uni eligibility). 4m for elevated reds.
    # 6m dropped — 3-stack reds are rare; the 4m sweep catches 2-stack reds.
    ALTITUDE_PHASES_M = [2.0, 4.0]
    PHASE_DURATIONS_S = [300.0, 200.0]   # 5 min ground sweep, ~3.3 min elevated sweep
    CLIMB_SPEED_M_S = 1.2
    ALTITUDE_TOLERANCE_M = 0.3
    # Locality-aware climbing: only climb when in open space (avoid altitude cycling in pockets)
    OPEN_SPACE_RADIUS_M = 3.0           # free clearance required around drone to allow climbing
    CLIMB_DEFER_TIMEOUT_S = 30.0        # if we keep deferring, climb anyway after this

    # Detection cadence
    DETECTION_BURST_FRAMES = 5            # baseline periodic burst (during forward motion)
    DETECTION_TRIGGER_INTERVAL_S = 2.0
    DETECTION_REACHED_BURST = 15          # bigger burst when arriving at a frontier
    DETECTION_SCAN_BURST = 12             # bigger burst after a 360° scan
    DETECTION_SKIP_LABELS = {"ROTATE_SAFE", "HOVER"}  # don't waste burst while wedged

    # Persistent escape — if N stales cluster in one area, ban the area entirely
    CLUSTER_RADIUS_M = 3.0
    CLUSTER_TRIGGER_COUNT = 3             # 3 stales within radius → ban this area

    # Control loop
    CONTROL_TICK_S = 0.1
    PLAN_INTERVAL_S = 0.3   # replan every N ticks at this period
    GRID_VISIT_RADIUS_M = 1.6   # marks drone's footprint + a bit — prevents "frontier inside me"

    # 360° scan cadence
    PERIODIC_SCAN_INTERVAL_S = 50.0
    SCAN_QUARTER_HOLD_S = 0.4
    SCAN_YAW_RATE_DEG_S = 90.0

    # Frontier handling
    FRONTIER_TIMEOUT_S = 20.0
    FRONTIER_REACH_M = 1.5      # was 1.0 — match cell size; avoids near-zero picks
    FRONTIER_MIN_DIST_M = 3.0   # ignore frontiers closer than this — forces outward exploration

    # Dead-end escape: if we blacklist this many frontiers within a short window, switch to "go far" mode
    DEAD_END_BLACKLIST_COUNT = 3
    DEAD_END_WINDOW_S = 45.0

    # Return-to-spawn after altitude climb — forces re-scan of main box at the new altitude
    RETURN_TARGET_NED = (2.0, 2.0)   # central-ish point inside main box
    RETURN_REACH_M = 3.0             # close enough to spawn area to call it done
    RETURN_TIMEOUT_S = 60.0          # bail if return takes too long (don't burn phase budget)

    MISSION_DURATION_S = 540.0   # 9 min; 1 min landing buffer

    def __init__(self, depth_topic="/depth_camera", rgb_topic=RGB_TOPIC, model_path="barrels_v2.pt"):
        super().__init__(depth_topic=depth_topic)
        self.detector = GZPhotoDetectorSaver(
            topic=rgb_topic,
            save_dir="output",
            model_path=model_path,
            threshold=0.35,   # lowered from 0.5 — catch low-confidence barrels at distance/angle; HSV filter rejects FPs
        )
        self.current_ned = None
        self.current_yaw_deg = None

        self.grid = CostGrid(GridConfig())
        self.dwa = DwaPlanner(self.grid, DwaConfig())
        self.frontier = FrontierPicker(self.grid, FrontierConfig(
            reach_tolerance_m=self.FRONTIER_REACH_M,
            min_distance_m=self.FRONTIER_MIN_DIST_M,
            timeout_s=self.FRONTIER_TIMEOUT_S,
            visit_radius_m=self.GRID_VISIT_RADIUS_M,
        ))
        self.goal_world = None      # (n, e) of current frontier goal
        self.goal_picked_at = 0.0
        self.recent_blacklists: list[float] = []  # timestamps of recent stale-frontier blacklists
        self.recent_stale_positions: list[tuple[float, float]] = []  # world coords of recent stales (cluster detection)
        self.current_motion_label = "FORWARD"  # last DWA label — used to gate detection bursts

    # ---------------- vision-drone readiness ----------------

    async def wait_until_ready(self, timeout_s=60.0):
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
            await self.drone.action.set_takeoff_altitude(self.ALTITUDE_PHASES_M[0])
        except Exception:
            pass
        print(f"Takeoff to {self.ALTITUDE_PHASES_M[0]:.1f}m")
        await self.drone.action.takeoff()
        async for pos in self.drone.telemetry.position():
            if float(pos.relative_altitude_m) >= self.ALTITUDE_PHASES_M[0] - 0.20:
                break
        print("Takeoff complete")
        await asyncio.sleep(2.0)

    # ---------------- background pollers ----------------

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
            # Skip the burst if we're currently wedged/spinning — we're looking at walls
            if self.current_motion_label not in self.DETECTION_SKIP_LABELS:
                self.detector.trigger_detection_burst(self.DETECTION_BURST_FRAMES)
            await asyncio.sleep(self.DETECTION_TRIGGER_INTERVAL_S)

    # ---------------- altitude / scan ----------------

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

    async def scan_360(self):
        print("  ↺ scan 360°")
        for _ in range(4):
            await self.yaw_in_place(self.SCAN_YAW_RATE_DEG_S, 90.0 / self.SCAN_YAW_RATE_DEG_S)
            # Integrate depth during the dwell so the new view enters the grid
            for _ in range(int(self.SCAN_QUARTER_HOLD_S / self.CONTROL_TICK_S)):
                await self.set_body_velocity(0.0, 0.0, 0.0, 0.0)
                await asyncio.sleep(self.CONTROL_TICK_S)
                self._integrate_one_frame()
        # 360° gives us the richest view of the local area — fire a bigger burst now
        self.detector.trigger_detection_burst(self.DETECTION_SCAN_BURST)

    async def _return_to_spawn_area(self):
        """Fly toward RETURN_TARGET_NED using DWA. Bails on timeout.

        Called after a phase-2 climb so the drone re-traverses main box at the new altitude
        (where elevated reds become visible). Detection bursts fire during the flight.
        """
        loop = asyncio.get_running_loop()
        tn, te = self.RETURN_TARGET_NED
        print(f"  ↩ returning to main box ({tn:.1f},{te:.1f}) for re-scan at altitude")
        start = loop.time()
        self.dwa.reset_commitment()
        last_burst = 0.0
        while self.running and (loop.time() - start) < self.RETURN_TIMEOUT_S:
            self._integrate_one_frame()
            if self.current_ned is None or self.current_yaw_deg is None:
                await asyncio.sleep(self.CONTROL_TICK_S)
                continue
            cn, ce = self.current_ned
            d = math.hypot(tn - cn, te - ce)
            if d < self.RETURN_REACH_M:
                print(f"  ↩ reached main box at ({cn:.1f},{ce:.1f})")
                self.detector.trigger_detection_burst(self.DETECTION_REACHED_BURST)
                return
            cmd = self.dwa.plan(cn, ce, self.current_yaw_deg, tn, te)
            self.current_motion_label = cmd.label
            await self.set_body_velocity(cmd.forward_m_s, 0.0, 0.0, cmd.yaw_rate_deg_s)
            # Periodic burst during the return — detect anything we fly over
            now = loop.time()
            if now - last_burst >= 2.0:
                if cmd.label not in self.DETECTION_SKIP_LABELS:
                    self.detector.trigger_detection_burst(self.DETECTION_BURST_FRAMES)
                last_burst = now
            await asyncio.sleep(self.CONTROL_TICK_S)
        print(f"  ↩ return timed out after {self.RETURN_TIMEOUT_S:.0f}s — resuming frontier explore from here")

    # ---------------- grid maintenance ----------------

    def _integrate_one_frame(self) -> bool:
        """Pull the latest depth frame, integrate into grid. Returns False if no frame yet."""
        if self.current_ned is None or self.current_yaw_deg is None:
            return False
        depth = self.receiver.get_frame()
        if depth is None:
            return False
        n, e = self.current_ned
        self.grid.recenter(n, e)
        self.grid.tick_decay()
        self.grid.mark_visited(n, e, self.GRID_VISIT_RADIUS_M)
        self.grid.integrate_depth(depth, n, e, self.current_yaw_deg)
        return True

    def _is_open_space(self) -> bool:
        """True if drone has free clearance >= OPEN_SPACE_RADIUS_M in all 4 cardinal directions.
        Used to decide whether to commit to an altitude change (which is costly in pockets).
        """
        if self.current_ned is None:
            return False
        n, e = self.current_ned
        r = self.OPEN_SPACE_RADIUS_M
        # Sample 8 cells in a ring; require all to be non-occupied
        import math as _m
        for ang_deg in range(0, 360, 45):
            a = _m.radians(ang_deg)
            sn, se = n + r * _m.cos(a), e + r * _m.sin(a)
            i, j = self.grid.world_to_cell(sn, se)
            if not self.grid.in_bounds(i, j):
                return False
            if self.grid.is_occupied(i, j):
                return False
        return True

    # ---------------- main control loop ----------------

    async def explore_loop(self):
        loop = asyncio.get_running_loop()
        mission_end = loop.time() + self.MISSION_DURATION_S
        phase_idx = 0
        phase_end = loop.time() + self.PHASE_DURATIONS_S[phase_idx]
        climb_pending_since = None   # set when phase elapsed but we're in a pocket
        last_scan = loop.time()
        last_plan = 0.0
        last_log = 0.0
        last_command = (0.0, 0.0)

        await self._climb_to_altitude(self.ALTITUDE_PHASES_M[phase_idx])

        # Prime the grid with a few frames before flying
        for _ in range(20):
            self._integrate_one_frame()
            await asyncio.sleep(self.CONTROL_TICK_S)

        while self.running and loop.time() < mission_end:
            now = loop.time()

            # Altitude phase transition — defer if drone is in a tight pocket
            if now >= phase_end and phase_idx + 1 < len(self.ALTITUDE_PHASES_M):
                in_open = self._is_open_space()
                if climb_pending_since is None:
                    climb_pending_since = now
                force_climb = (now - climb_pending_since) >= self.CLIMB_DEFER_TIMEOUT_S
                if in_open or force_climb:
                    phase_idx += 1
                    phase_end = now + self.PHASE_DURATIONS_S[phase_idx]
                    climb_pending_since = None
                    reason = "open space" if in_open else "deferred timeout"
                    print(f"\n=== ALTITUDE PHASE {phase_idx + 1}/{len(self.ALTITUDE_PHASES_M)}: "
                          f"{self.ALTITUDE_PHASES_M[phase_idx]:.1f}m ({reason}) ===")
                    await self._climb_to_altitude(self.ALTITUDE_PHASES_M[phase_idx])
                    # Reset visited-cell memory so frontier picker re-explores the whole map at the new altitude.
                    # Without this, the main box (visited at 2m) would never be re-scanned at 4m → elevated reds missed.
                    self.grid.visited[:, :] = False
                    self.frontier.clear_blacklist()
                    self.recent_blacklists.clear()
                    self.recent_stale_positions.clear()
                    self.goal_world = None
                    print("  ↻ visited grid + blacklists cleared — re-explore at new altitude")
                    # Fly back to main box at new altitude — picker alone won't pull drone south from N corridor
                    await self._return_to_spawn_area()
                    last_scan = loop.time()
                    continue

            # Periodic 360° scan
            if now - last_scan >= self.PERIODIC_SCAN_INTERVAL_S:
                await self.scan_360()
                last_scan = loop.time()
                continue

            # Always integrate the freshest depth view
            self._integrate_one_frame()

            # Frontier management
            if self.current_ned is not None:
                cn, ce = self.current_ned
                if self.goal_world is not None:
                    gn, ge = self.goal_world
                    d = math.hypot(gn - cn, ge - ce)
                    stale = (now - self.goal_picked_at) > self.FRONTIER_TIMEOUT_S
                    if d < self.FRONTIER_REACH_M or stale:
                        if stale:
                            print(f"  ✗ frontier stale, blacklisting ({gn:.1f},{ge:.1f})")
                            self.frontier.blacklist_cell(gn, ge)
                            self.recent_blacklists.append(now)
                            self.recent_stale_positions.append((gn, ge))
                            # Cluster check: if too many stales fell inside CLUSTER_RADIUS_M, ban this whole area
                            nearby = [p for p in self.recent_stale_positions
                                      if math.hypot(p[0] - gn, p[1] - ge) <= self.CLUSTER_RADIUS_M]
                            if len(nearby) >= self.CLUSTER_TRIGGER_COUNT:
                                cn0 = sum(p[0] for p in nearby) / len(nearby)
                                ce0 = sum(p[1] for p in nearby) / len(nearby)
                                # Blacklist a bigger neighbourhood around the cluster centroid
                                for r_step in (0.0, 0.6, 1.2, 1.8, 2.4):
                                    for ang in range(0, 360, 30):
                                        self.frontier.blacklist_cell(
                                            cn0 + r_step * math.cos(math.radians(ang)),
                                            ce0 + r_step * math.sin(math.radians(ang)),
                                        )
                                print(f"  ⊘ cluster ban around ({cn0:.1f},{ce0:.1f}) "
                                      f"after {len(nearby)} nearby stales")
                                # Drop these stale entries so we don't re-trigger immediately
                                self.recent_stale_positions = [p for p in self.recent_stale_positions
                                                                if p not in nearby]
                        else:
                            print(f"  ✓ reached frontier ({gn:.1f},{ge:.1f})")
                            # Fire a long burst — we deliberately came here, look hard
                            self.detector.trigger_detection_burst(self.DETECTION_REACHED_BURST)
                        self.goal_world = None
                if self.goal_world is None:
                    # Trim old blacklist timestamps outside the dead-end window
                    cutoff = now - self.DEAD_END_WINDOW_S
                    self.recent_blacklists = [t for t in self.recent_blacklists if t >= cutoff]
                    escape = len(self.recent_blacklists) >= self.DEAD_END_BLACKLIST_COUNT
                    new_goal = self.frontier.pick(cn, ce, self.current_yaw_deg, prefer_far=escape)
                    if new_goal is not None:
                        self.goal_world = new_goal
                        self.goal_picked_at = now
                        self.dwa.reset_commitment()
                        tag = " [ESCAPE]" if escape else ""
                        print(f"  → frontier ({new_goal[0]:.1f},{new_goal[1]:.1f}){tag}")
                        if escape:
                            self.recent_blacklists.clear()  # reset after committing to escape

            # Replan periodically; otherwise hold last command
            if now - last_plan >= self.PLAN_INTERVAL_S and self.current_ned is not None and self.current_yaw_deg is not None:
                gn, ge = self.goal_world if self.goal_world else (None, None)
                cmd = self.dwa.plan(self.current_ned[0], self.current_ned[1],
                                    self.current_yaw_deg, gn, ge)
                last_command = (cmd.forward_m_s, cmd.yaw_rate_deg_s)
                self.current_motion_label = cmd.label
                last_plan = now
                if now - last_log >= 1.0:
                    last_log = now
                    remaining = int(mission_end - now)
                    goal_str = f"({gn:.1f},{ge:.1f})" if gn is not None else "none"
                    print(f"{cmd.label} v={cmd.forward_m_s:.2f} w={cmd.yaw_rate_deg_s:+.0f}°/s "
                          f"goal={goal_str} t-{remaining}s")

            # Push command to drone (steady velocity at CONTROL_TICK_S resolution)
            v, w = last_command
            await self.set_body_velocity(v, 0.0, 0.0, w)
            await asyncio.sleep(self.CONTROL_TICK_S)

        print("\nMission time elapsed — landing")

    async def run(self):
        print("Starting RoboVerse Qualifier mission (cost-grid + DWA + frontier)")
        bg_tasks = []
        try:
            bg_tasks.append(asyncio.create_task(self.detector.run()))
            bg_tasks.append(asyncio.create_task(self._trigger_detection_loop()))

            await self.connect()
            bg_tasks.append(asyncio.create_task(self._poll_position()))
            bg_tasks.append(asyncio.create_task(self._poll_attitude()))

            await self.arm_and_takeoff()
            await self.start_offboard()
            await self.explore_loop()

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
