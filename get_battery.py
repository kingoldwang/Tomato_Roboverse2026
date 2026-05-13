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

    print("Subscribing to battery (Ctrl+C to stop)...")
    try:
        async for battery in drone.telemetry.battery():
            print(
                f"\rVoltage: {battery.voltage_v:5.2f} V  "
                f"Remaining: {battery.remaining_percent:6.2f} %",
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
