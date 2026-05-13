from mavsdk import System
from mavsdk.offboard import Offboard
from mavsdk.offboard import VelocityNedYaw, PositionNedYaw
from mavsdk.telemetry import LandedState
import asyncio
import math

class Drone:
    def __init__(self):
        self.drone = System()

    def _normalize_yaw(self, yaw_deg):
        while yaw_deg > 180:
            yaw_deg -= 360
        while yaw_deg < -180:
            yaw_deg += 360
        return yaw_deg

    def _yaw_error(self, target, current):
        error = target - current
        while error > 180:
            error -= 360
        while error < -180:
            error += 360
        return error

    async def connect(self):
        await self.drone.connect(system_address="udpin://0.0.0.0:14540")

        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("Connected")
                break

        # Disable simulated battery drain (SITL only)
        await self.drone.param.set_param_float("SIM_BAT_MIN_PCT", 99.0)

        print("Waiting for EKF / global position...")
        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print("EKF ready")
                break

    async def arm_and_takeoff(self, target_altitude=2.5, timeout=30.0):
        await self.drone.action.arm()
        await self.drone.action.takeoff()

        async def _wait_altitude():
            async for pos in self.drone.telemetry.position():
                if pos.relative_altitude_m >= target_altitude - 0.2:
                    return

        try:
            await asyncio.wait_for(_wait_altitude(), timeout=timeout)
        except asyncio.TimeoutError:
            raise RuntimeError(f"Takeoff did not reach {target_altitude} m within {timeout}s")

        print("Takeoff")
        await self.drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
        await self.drone.offboard.start()

    async def land(self, timeout=30.0):
        await self.drone.offboard.stop()
        await self.drone.action.land()

        async def _wait_landed():
            async for landed in self.drone.telemetry.landed_state():
                if landed == LandedState.ON_GROUND:
                    return

        try:
            await asyncio.wait_for(_wait_landed(), timeout=timeout)
        except asyncio.TimeoutError:
            print(f"Land timed out after {timeout}s, forcing disarm")

        print("land")
        await self.drone.action.disarm()

    async def get_position(self):
        async for pos in self.drone.telemetry.position_velocity_ned():
            return pos.position.north_m, pos.position.east_m, pos.position.down_m

    async def get_yaw(self):
        async for att in self.drone.telemetry.attitude_euler():
            return att.yaw_deg

    async def send_velocity(self, vx, vy, vz,yaw_deg):
         await self.drone.offboard.set_velocity_ned(VelocityNedYaw(north_m_s=vx, east_m_s=vy, down_m_s=vz, yaw_deg=yaw_deg))

    async def send_position_setpoint(self, north, east, down, yaw_deg):
        await self.drone.offboard.set_position_ned(PositionNedYaw(north_m=north, east_m=east, down_m=down, yaw_deg=yaw_deg))

    async def rotate_to_yaw(self, target_yaw_deg, tolerance=2.0):
        """
        Rotate to a target yaw using PID control
        """
        target_yaw_deg = self._normalize_yaw(target_yaw_deg)

        # PID gains (tune these!)
        Kp = 0.8
        Ki = 0.0
        Kd = 0.2

        integral = 0.0
        prev_error = 0.0

        dt = 0.1  # 10 Hz loop

        while True:
#            yaw_rad = await self.get_yaw()
            current_yaw = await self.get_yaw()

            error = self._yaw_error(target_yaw_deg, current_yaw)

            # Stop condition
            if abs(error) < tolerance:
                break

            # PID terms
            integral += error * dt
            derivative = (error - prev_error) / dt

            output = Kp * error + Ki * integral + Kd * derivative

            # Clamp yaw rate (deg/s equivalent behavior)
            max_yaw_rate = 60.0
            output = max(min(output, max_yaw_rate), -max_yaw_rate)

            # Convert to target yaw step
            new_yaw = current_yaw + output * dt
            new_yaw = self._normalize_yaw(new_yaw)

            # Send command
            await self.drone.offboard.set_velocity_ned(
                VelocityNedYaw(
                    north_m_s=0.0,
                    east_m_s=0.0,
                    down_m_s=0.0,
                    yaw_deg=new_yaw
                )
            )

            prev_error = error
            await asyncio.sleep(dt)

        # Final stabilization
        await self.drone.offboard.set_velocity_ned(
            VelocityNedYaw(0.0, 0.0, 0.0, target_yaw_deg)
        )

    # =========================
    # 🚁 HIGH-LEVEL COMMANDS
    # =========================

    async def turn_cw_90(self):
        current = await self.get_yaw()
        await self.rotate_to_yaw(current + 90)

    async def turn_ccw_90(self):
        current = await self.get_yaw()
        await self.rotate_to_yaw(current - 90)

    async def turn_cw_180(self):
        current = await self.get_yaw()
        await self.rotate_to_yaw(current + 180)