#!/usr/bin/env python3
import asyncio
import numpy as np
import time

from depth_receiver import DepthReceiver
from drone_control import Drone
from AvoidancePlanner import AvoidancePlanner
from get_position_with_task import SharedState, position_monitor_task

class DroneNavigation:
    def __init__(self,
                 depth_topic="/depth_camera",
                 loop_hz=20.0,
                 max_runtime_s=60.0):

        self.loop_hz = loop_hz
        self.max_runtime_s = max_runtime_s
        self.running = True

        # =========================
        # GRID HEADING SYSTEM
        # =========================
        self.grid_headings = [0, 90, 180, -90]  # N, E, S, W
        self.current_heading_idx = 0
        self.target_yaw_deg = self.grid_headings[self.current_heading_idx]
        self.yaw_tolerance = 5.0

        # =========================
        #  NED POSE TRACKING
        # =========================
        self.pose = {
            "north": 0.0,
            "east": 0.0,
            "down": -2.0,
            "yaw": 0.0,
            "yaw_deg": 0.0
        }

        # Camera intrinsics
        K = np.array([[433.0, 0.0, 320.0],
                      [0.0, 433.0, 240.0],
                      [0.0, 0.0, 1.0]])

        self.receiver = DepthReceiver(depth_topic)

        self.planner = AvoidancePlanner(
            K=K,
            width=640,
            height=480,
            safe_distance=4.0,
            critical_distance=1.5
        )

        self.drone = Drone()
        self.position_state = SharedState()    


    # =========================
    #  YAW UTILS
    # =========================
    def _yaw_error(self, target, current):
        error = target - current
        while error > 180:
            error -= 360
        while error < -180:
            error += 360
        return error

    async def update_pose(self):
        """
        Get pose from drone from state shared by position monitor task. This is critical to ensure the planner has the latest pose for decision making.
        """
        self.pose["north"] = self.position_state.latest_position.north_m
        self.pose["east"]  = self.position_state.latest_position.east_m
        self.pose["down"]  = self.position_state.latest_position.down_m
        self.pose["yaw_deg"]   = self.position_state.latest_yaw
        self.pose["yaw"] = np.deg2rad(self.pose["yaw_deg"])

    async def align_to_grid(self):
        current_yaw = await self.drone.get_yaw()
        error = self._yaw_error(self.target_yaw_deg, current_yaw)

        if abs(error) > self.yaw_tolerance:
            print(f"Aligning to {self.target_yaw_deg}° (err={error:.2f})")
            await self.drone.rotate_to_yaw(self.target_yaw_deg)

    # =========================
    # 🔄 GRID TURNING
    # =========================
    async def rotate_next_direction(self):
        self.current_heading_idx = (self.current_heading_idx + 1) % 4
        self.target_yaw_deg = self.grid_headings[self.current_heading_idx]

        print(f"🔄 New heading: {self.target_yaw_deg}°")
        await self.drone.rotate_to_yaw(self.target_yaw_deg)

    # =========================
    # tHE MAIN LOOP WHERE THE PIPELINE COMES TOGETHER
    # =========================
    async def _wait_for_first_data(self, timeout=10.0):
        """Block until position monitor and depth receiver have their first frame."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if (self.position_state.latest_position is not None
                and self.position_state.latest_yaw is not None
                and self.receiver.get_frame() is not None):
                return
            await asyncio.sleep(0.1)
        raise RuntimeError("Timed out waiting for first position/depth data")

    async def run(self):
        print("\nPOSITION-BASED AUTONOMOUS AvoidanceNAVIGATION\n")

        await self.drone.connect()
        print("Starting position monitor.")
        self._monitor_stop = asyncio.Event()
        self.monitor_task = asyncio.create_task(
            position_monitor_task(self.drone, self.position_state, self._monitor_stop)
        )
        await self.drone.arm_and_takeoff()

        print("Waiting for first position + depth frame...")
        await self._wait_for_first_data()

        # Initial alignment
        await self.drone.rotate_to_yaw(self.target_yaw_deg)

        t_run_start = time.monotonic()

        try:
            while self.running:
                t_start = time.monotonic()

                if t_start - t_run_start > self.max_runtime_s:
                    print(f"\nMax runtime {self.max_runtime_s}s reached, ending flight.")
                    break

                await self.update_pose()

                depth_frame = self.receiver.get_frame()
                if depth_frame is None:
                    await asyncio.sleep(1.0 / self.loop_hz)
                    continue

                north, east, down, info = self.planner.compute_position_ned(
                    depth_frame,
                    self.pose,
                    step_size=1.5
                )

                c = info['clearance']

                print(f"Blocked={info['blocked']} | "
                      f"Target N={north:.2f}, E={east:.2f} | "
                      f"L={c['left']:.2f} C={c['center']:.2f} R={c['right']:.2f}")

                if info['blocked']:
                    await self.drone.send_velocity(0, 0, 0, self.target_yaw_deg)
                    await self.rotate_next_direction()
                else:
                    await self.align_to_grid()
                    await self.drone.send_position_setpoint(
                        north=north,
                        east=east,
                        down=down,
                        yaw_deg=self.target_yaw_deg
                    )

                elapsed = time.monotonic() - t_start
                sleep_time = (1.0 / self.loop_hz) - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            print("Navigation cancelled")

        finally:
            print("Landing...")
            try:
                await self.drone.land()
            except Exception as e:
                print(f"Land failed: {e}")
            self._monitor_stop.set()
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except (asyncio.CancelledError, Exception):
                pass
            print("Shutdown complete.")

    def stop(self):
        self.running = False


# =========================
#  ENTRY POINT
# =========================
async def main():
    nav = DroneNavigation()

    task = asyncio.create_task(nav.run())

    try:
        await task
    except KeyboardInterrupt:
        print("\n⌨️ Stopping...")
        nav.stop()
        await asyncio.gather(task, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())