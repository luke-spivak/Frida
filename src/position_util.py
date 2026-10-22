"""Read brush-tip XYZ in FRIDA coordinates without changing motor registers.

Run from src: python position_util.py
Torque stays in its existing state, including when exiting.
"""

from robot import SO101
from preview_so101_offsets import ReadOnlyFeetechBus


def main():
    robot = SO101(debug=True, show_viewer=False)
    # Reuse the follower's calibrated degree normalization, but block writes.
    robot.arm.bus = ReadOnlyFeetechBus(
        robot.arm.config.port, robot.arm.bus.motors,
        calibration=robot.arm.calibration,
    )
    bus = robot.arm.bus

    try:
        # Follower.connect()/good_morning_robot() configure motors and toggle
        # torque. The bus handshake only pings and reads motor registers.
        bus.connect()
        if not bus.is_calibrated:
            raise RuntimeError(
                "Saved calibration does not match motor registers. "
                "No registers were changed."
            )
        print("Read-only brush-tip positions in FRIDA coordinates (meters). Torque is unchanged.")
        while True:
            label = input(
                "Enter a pose label to read the current position (or q to quit): "
            ).strip()
            if label.lower() == "q":
                break

            print({
                "label": label,
                "frida_xyz_m": robot.get_cartesian_pose(),
            })
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()

# Previous reference measurements:
# Water [-0.2523272205447978, 0.1279365474405061, 0.05768630136705648]
# Towel [-0.23466353444669566, 0.030717544685104438, 0.06278398806598728]
# Paint [-0.25458134738718924, 0.2238353050883219, 0.052609362291599494]

# [-0.24000476864733444, 0.03195579133333619, 0.06153459122646422]
# [-0.22836268170595586, 0.0037117373507086146, 0.023758581702215467]
# [-0.2284859094869825, 0.10288259464919307, 0.09153689456573225]

# New rage: [-0.19836798817407578, 0.08626855950381895, -0.023568702248490905]
# New water: [-0.20184177295895028, 0.16764979740156266, -0.029987460665286153]
# New paint: [[-0.199800221679217, 0.24693064409612758, -0.034320224952209974]]
# Canvas: [-0.016076763833475734, 0.22854904644058543, -0.025097812407582396]

# Attitude/rotvec [-1.2958252302262085, 1.2223351981677575, -1.0658945719158899]

# Neutral position: [-0.1758190516059921, 0.12804887593285125, 0.04232101833406253]
