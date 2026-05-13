import asyncio
from mavsdk import System

async def run():
    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    print("Waiting for PX4 connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected to PX4")
            break

    print("Subscribing to position_velocity_ned (Ctrl+C to stop)...")
    try:
        async for pos_vel in drone.telemetry.position_velocity_ned():
            p = pos_vel.position
            v = pos_vel.velocity
            print(
                f"\rN: {p.north_m:8.3f}m  E: {p.east_m:8.3f}m  D: {p.down_m:8.3f}m  "
                f"vN: {v.north_m_s:6.2f}  vE: {v.east_m_s:6.2f}  vD: {v.down_m_s:6.2f}",
                end="",
                flush=True,
            )
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nStopped.")
