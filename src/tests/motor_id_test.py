
import time
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

robot = SO101Follower(SO101FollowerConfig(
    port="/dev/tty.usbmodem5B8E1155331",
    id="my_arm",
    use_degrees=True,
    max_relative_target=5.0,
))

try:
    robot.connect()

    motors = sorted(
        robot.bus.motors.items(),
        key=lambda item: item[1].id
    )

    print("\nMotor ID mapping:")
    for name, motor in motors:
        print(f"ID {motor.id}: {name}")

    print("\nEnsure the arm is supported and has clearance.")
    input("Press Enter to begin...")

    for name, motor in motors:
        key = f"{name}.pos"
        delta = 5.0 if name == "gripper" else 3.0

        print(f"\n--- Motor {motor.id}: {name} ---")

        choice = input(
            "Enter = move positive, '-' = negative, "
            "'s' = skip, 'q' = quit: "
        ).strip().lower()

        if choice == "q":
            break
        if choice == "s":
            continue

        direction = -1 if choice == "-" else 1

        # Record current joint position
        start = float(robot.get_observation()[key])
        target = start + direction * delta

        print(f"Moving {name}: {start:.2f} -> {target:.2f}")

        # Move only this motor
        robot.send_action({key: target})
        time.sleep(1.0)

        actual = float(robot.get_observation()[key])
        print(f"Measured position: {actual:.2f}")

        input("Verify the joint, then press Enter to return...")

        # Return to starting position
        robot.send_action({key: start})
        time.sleep(1.0)

        returned = float(robot.get_observation()[key])
        print(f"Returned to: {returned:.2f}")

    print("\nMotor ID test finished.")

except KeyboardInterrupt:
    print("\nTest interrupted.")

finally:
    if robot.is_connected:
        robot.disconnect()
