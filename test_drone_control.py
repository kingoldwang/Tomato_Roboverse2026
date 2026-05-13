"""
Step 3 — End-to-end test of the Drone() wrapper in drone_control.py.

Exercises: arm_and_takeoff, get_yaw, get_position, rotate_to_yaw,
send_position_setpoint, send_velocity, land.

The drone takes off, rotates through 90 / 180 / 270 / 0, flies 2 m north,
hovers, then lands.
"""

import asyncio
from drone_control import Drone

HOVER_HZ = 20
HOVER_DT = 1.0 / HOVER_HZ


async def hover_for(drone: Drone, seconds: float, yaw_deg: float | None = None):
    """Keep offboard alive with zero-velocity setpoints at HOVER_HZ."""
    if yaw_deg is None:
        yaw_deg = await drone.get_yaw()
    for _ in range(int(seconds * HOVER_HZ)):
        await drone.send_velocity(0.0, 0.0, 0.0, yaw_deg)
        await asyncio.sleep(HOVER_DT)


async def fly_to(drone: Drone, north: float, east: float, down: float,
                 yaw_deg: float, seconds: float):
    """Stream a position setpoint at HOVER_HZ for `seconds`."""
    for _ in range(int(seconds * HOVER_HZ)):
        await drone.send_position_setpoint(north, east, down, yaw_deg)
        await asyncio.sleep(HOVER_DT)


async def main():
    drone = Drone()
    await drone.connect()
    await drone.arm_and_takeoff()

    print("\n[TEST] Stabilising hover (3 s)")
    await hover_for(drone, 3.0)

    n0, e0, d0 = await drone.get_position()
    yaw0 = await drone.get_yaw()
    print(f"[TEST] Pose after takeoff: N={n0:.2f} E={e0:.2f} D={d0:.2f} yaw={yaw0:.1f}")

    print("\n[TEST] Rotate CW 90")
    await drone.turn_cw_90()
    await hover_for(drone, 1.0)
    print(f"[TEST] yaw={await drone.get_yaw():.1f}")

    print("\n[TEST] Rotate CW 180")
    await drone.turn_cw_180()
    await hover_for(drone, 1.0)
    print(f"[TEST] yaw={await drone.get_yaw():.1f}")

    print("\n[TEST] Rotate CCW 90 (back toward north)")
    await drone.turn_ccw_90()
    await hover_for(drone, 1.0)
    print(f"[TEST] yaw={await drone.get_yaw():.1f}")

    print("\n[TEST] Fly +2 m north, hold 5 s")
    yaw_now = await drone.get_yaw()
    await fly_to(drone, n0 + 2.0, e0, d0, yaw_now, seconds=5.0)
    n1, e1, d1 = await drone.get_position()
    print(f"[TEST] Pose after waypoint: N={n1:.2f} E={e1:.2f} D={d1:.2f}")

    print("\n[TEST] Landing")
    await drone.land()
    print("[TEST] Done.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
