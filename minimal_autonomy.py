#!/usr/bin/env python3
import asyncio
import sys
import time

import numpy as np
from mavsdk import System, telemetry
from mavsdk.action import ActionError
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

from depth_receiver import DepthReceiver


MAVSDK_ADDRESS = "udpin://0.0.0.0:14540"
TAKEOFF_ALTITUDE_M = 2.0
ENABLE_DEBUG = True


class MinimalAutonomy:
    """
    Simple student-facing autonomy demo.

    Lifecycle:
    connect -> wait ready -> arm -> take off -> start offboard -> run until stopped
    -> stop motion -> stop offboard -> land -> wait landed -> disarm.
    """

    SAFE_DISTANCE_M = 2.0
    CRITICAL_DISTANCE_M = 0.3
    DECISION_PERIOD_S = 1.0
    TURN_DEGREES = 90.0
    FORWARD_SPEED_M_S = 0.8
    YAW_RATE_DEG_S = 45.0
    CONTROL_DT_S = 0.1
    ROI_TOP_FRACTION = 0.25
    ROI_BOTTOM_FRACTION = 0.70

    def __init__(self, depth_topic="/depth_camera"):
        self.depth_topic = depth_topic
        self.receiver = DepthReceiver(depth_topic)
        self.drone = System()
        self.running = True
        self.offboard_started = False
        self.shutdown_started = False

    def _valid_depth_values(self, region):
        region = np.asarray(region, dtype=np.float32)
        return region[np.isfinite(region) & (region > 0.0)]

    def _sector_clearance(self, region):
        valid = self._valid_depth_values(region)
        if valid.size == 0:
            return 0.0
        return float(np.percentile(valid, 20))

    def compute_clearances(self, depth_frame):
        height, width = depth_frame.shape
        row_start = int(height * self.ROI_TOP_FRACTION)
        row_end = int(height * self.ROI_BOTTOM_FRACTION)
        roi = depth_frame[row_start:row_end, :]

        split = width // 3
        left = self._sector_clearance(roi[:, :split])
        center = self._sector_clearance(roi[:, split: 2 * split])
        right = self._sector_clearance(roi[:, 2 * split:])
        return left, center, right

    def decide_motion(self, left, center, right):
        if center >= self.SAFE_DISTANCE_M:
            return "FORWARD", self.DECISION_PERIOD_S

        if (
            left < self.CRITICAL_DISTANCE_M
            and center < self.CRITICAL_DISTANCE_M
            and right < self.CRITICAL_DISTANCE_M
        ):
            return "STOP", self.DECISION_PERIOD_S

        if left >= right:
            return "TURN_LEFT", self.TURN_DEGREES / self.YAW_RATE_DEG_S
        return "TURN_RIGHT", self.TURN_DEGREES / self.YAW_RATE_DEG_S

    async def connect(self):
        print(f"Connecting to PX4 SITL on {MAVSDK_ADDRESS} ...")
        await self.drone.connect(system_address=MAVSDK_ADDRESS)
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("Connected")
                break

    async def wait_until_ready(self, timeout_s=30.0):
        print("Waiting for vehicle readiness...")
        loop = asyncio.get_running_loop()
        start = loop.time()

        async for health in self.drone.telemetry.health():
            armable = getattr(health, "is_armable", False)
            local_ok = getattr(health, "is_local_position_ok", False)
            global_ok = getattr(health, "is_global_position_ok", False)
            home_ok = getattr(health, "is_home_position_ok", False)

            print(
                f"Health: armable={armable}, local_ok={local_ok}, "
                f"global_ok={global_ok}, home_ok={home_ok}"
            )

            if armable or local_ok or (global_ok and home_ok):
                print("Ready for takeoff!")
                return

            if loop.time() - start > timeout_s:
                raise TimeoutError("Timed out waiting for vehicle readiness")

    async def arm_and_takeoff(self):
        await self.wait_until_ready()

        print("Arming...")
        try:
            await self.drone.action.arm()
        except ActionError as e:
            raise RuntimeError(f"Arm failed: {e}") from e

        try:
            await self.drone.action.set_takeoff_altitude(TAKEOFF_ALTITUDE_M)
        except Exception:
            pass

        print(f"Takeoff to {TAKEOFF_ALTITUDE_M:.1f} m")
        try:
            await self.drone.action.takeoff()
        except ActionError as e:
            raise RuntimeError(f"Takeoff failed: {e}") from e

        async for pos in self.drone.telemetry.position():
            alt = float(pos.relative_altitude_m)
            sys.stdout.write(
                f"\rTakeoff altitude: {alt:.2f} / {TAKEOFF_ALTITUDE_M:.2f} m   "
            )
            sys.stdout.flush()
            if alt >= TAKEOFF_ALTITUDE_M - 0.20:
                break

        print("\nTakeoff complete")
        await asyncio.sleep(2.0)

    async def set_body_velocity(self, forward_m_s, right_m_s, down_m_s, yaw_rate_deg_s):
        await self.drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(
                forward_m_s=float(forward_m_s),
                right_m_s=float(right_m_s),
                down_m_s=float(down_m_s),
                yawspeed_deg_s=float(yaw_rate_deg_s),
            )
        )

    async def start_offboard(self):
        if self.offboard_started:
            return

        await self.set_body_velocity(0.0, 0.0, 0.0, 0.0)
        try:
            await self.drone.offboard.start()
        except OffboardError as e:
            raise RuntimeError(f"Offboard start failed: {e}") from e

        self.offboard_started = True
        print("Offboard started")

    async def hold_position(self, duration_s):
        steps = max(1, int(duration_s / self.CONTROL_DT_S))
        for _ in range(steps):
            await self.set_body_velocity(0.0, 0.0, 0.0, 0.0)
            await asyncio.sleep(self.CONTROL_DT_S)

    async def fly_forward(self, duration_s):
        steps = max(1, int(duration_s / self.CONTROL_DT_S))
        for _ in range(steps):
            await self.set_body_velocity(self.FORWARD_SPEED_M_S, 0.0, 0.0, 0.0)
            await asyncio.sleep(self.CONTROL_DT_S)

    async def yaw_in_place(self, yaw_rate_deg_s, duration_s):
        steps = max(1, int(duration_s / self.CONTROL_DT_S))
        for _ in range(steps):
            await self.set_body_velocity(0.0, 0.0, 0.0, yaw_rate_deg_s)
            await asyncio.sleep(self.CONTROL_DT_S)
        await self.hold_position(0.3)

    async def task_loop(self):
        print("Autonomy task loop started")
        while self.running:
            depth_frame = self.receiver.get_frame()
            if depth_frame is None:
                print("STOP | waiting for depth frame")
                await self.hold_position(self.DECISION_PERIOD_S)
                continue

            left, center, right = self.compute_clearances(depth_frame)
            action, duration_s = self.decide_motion(left, center, right)

            print(f"{action} | L={left:.2f} C={center:.2f} R={right:.2f}")

            if action == "FORWARD":
                if ENABLE_DEBUG:
                    print(
                        f"  body_forward={self.FORWARD_SPEED_M_S:.2f} "
                        f"duration={duration_s:.2f}"
                    )
                await self.fly_forward(duration_s)
                continue

            if action == "TURN_LEFT":
                if ENABLE_DEBUG:
                    print(
                        f"  yaw_delta=-{self.TURN_DEGREES:.1f} "
                        f"yaw_rate=-{self.YAW_RATE_DEG_S:.1f}"
                    )
                await self.yaw_in_place(-self.YAW_RATE_DEG_S, duration_s)
                continue

            if action == "TURN_RIGHT":
                if ENABLE_DEBUG:
                    print(
                        f"  yaw_delta={self.TURN_DEGREES:.1f} "
                        f"yaw_rate={self.YAW_RATE_DEG_S:.1f}"
                    )
                await self.yaw_in_place(self.YAW_RATE_DEG_S, duration_s)
                continue

            await self.hold_position(duration_s)

    async def stop_offboard(self):
        if not self.offboard_started:
            return

        print("Stopping offboard")
        try:
            await self.hold_position(0.3)
            await self.drone.offboard.stop()
        except Exception as e:
            print(f"Offboard stop skipped or failed: {e}")
        self.offboard_started = False

    async def land_and_wait(self):
        print("Landing")
        try:
            await self.drone.action.land()
        except Exception as e:
            print(f"Landing skipped or failed: {e}")
            return

        async for landed in self.drone.telemetry.landed_state():
            if landed == telemetry.LandedState.ON_GROUND:
                print("Landed")
                break

        try:
            await self.drone.action.disarm()
            print("Disarmed")
        except Exception as e:
            print(f"Disarm skipped or failed: {e}")

    async def shutdown(self):
        if self.shutdown_started:
            return
        self.shutdown_started = True

        self.running = False
        await self.stop_offboard()
        await self.land_and_wait()

    async def run(self):
        print("Starting minimal baseline autonomy")
        print(f"DEBUG={'ON' if ENABLE_DEBUG else 'OFF'}")

        try:
            await self.connect()
            await self.arm_and_takeoff()
            await self.start_offboard()
            await self.task_loop()
        except asyncio.CancelledError:
            print("Minimal autonomy cancelled")
            raise
        finally:
            await self.shutdown()

    def stop(self):
        self.running = False


async def main():
    controller = MinimalAutonomy()
    try:
        await controller.run()
    except KeyboardInterrupt:
        controller.stop()
        await controller.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
