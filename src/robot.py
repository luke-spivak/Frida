#! /usr/bin/env python
import os
from collections import deque
import numpy as np
import time


class MotionPlanningError(RuntimeError):
    """A Cartesian trajectory could not be planned; execution has not started."""


##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

class Robot:
    '''
        Low-level action functionality of the robot.
        This is an abstract class, see its children for usage.
    '''
    def __init__(self, debug):
        self.debug_bool = debug

    def debug(self, msg):
        if self.debug_bool:
            print(msg)

    def good_morning_robot(self):
        raise Exception("This method must be implemented")

    def good_night_robot(self):
        raise Exception("This method must be implemented")

    def go_to_cartesian_pose(self, positions, orientations, precise=False):
        raise Exception("This method must be implemented")

        
class SimulatedRobot(Robot, object):
    def __init__(self, debug=True):
        pass

    def good_morning_robot(self):
        pass

    def good_night_robot(self):
        pass

    def go_to_cartesian_pose(self, position, orientation):
        pass

        
"""Imports for SO-101"""
from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower
from motion_planner import SO101MotionPlanner

class SO101(Robot, object):
    '''
        Low-level action functionality of the SO-101 robot.
    '''
    def __init__(self, debug, show_viewer=False):
        self.debug_bool = debug
        
        # Config for arm
        config = SO101FollowerConfig(
            port="/dev/tty.usbmodem5B8E1155331",
            id="my_arm",
            use_degrees=True,
            # cameras={"top": 0, "side": 1},
            max_relative_target=None, # Limit max motion to 10 degrees
        )
        self.arm = SO101Follower(config=config)
        
        self.motor_names = list(self.arm.bus.motors.keys())
        self.arm_joint_names = [n for n in self.motor_names if n != "gripper"]
        self._last_arm_command_deg = None
        self.motion_planner = SO101MotionPlanner(
            show_viewer=show_viewer, lerobot_calibration_path=self.arm.calibration_fpath,
        )

    def debug(self, msg):
        if self.debug_bool:
            print(msg)

    def good_morning_robot(self):
        self._last_arm_command_deg = None
        self.arm.connect()
        
        with self.arm.bus.torque_disabled():
            self.arm.bus.write("P_Coefficient", "shoulder_lift", 88)
            self.arm.bus.write("P_Coefficient", "elbow_flex", 56)
            self.arm.bus.write("P_Coefficient", "shoulder_pan", 48)
            self.arm.bus.write("P_Coefficient", "wrist_flex", 56)
            self.arm.bus.write("P_Coefficient", "wrist_roll", 48)

        # Session settings, applied after LeRobot's connection/configuration.
        # Keep EEPROM locked, matching the successful dead-zone experiments.
        for name in self.arm_joint_names:
            for register in ("CW_Dead_Zone", "CCW_Dead_Zone"):
                self.arm.bus.write(register, name, 0, normalize=False)
                if self.arm.bus.read(register, name, normalize=False) != 0:
                    raise RuntimeError(f"Failed to set {name} {register} to zero")
            
    def good_night_robot(self):
        self._last_arm_command_deg = None
        self.arm.disconnect()
        
    @staticmethod
    def position_equal(target, actual, tolerance=1.0) -> bool:
        return all(
            abs(float(actual[k]) - float(v)) <= tolerance
            for k, v in target.items()
            if k.endswith(".pos")
        )
    
    def frida_to_robot(self, positions: np.ndarray) -> np.ndarray:
        # Transformation matrix mapping [x, y, z] -> [y, -x, z]
        transform = np.array([
            [ 0, -1,  0],  # New X = Old Y
            [ 1,  0,  0],  # New Y = -Old X
            [ 0,  0,  1],  # New Z = Old Z
        ])
        return positions @ transform
    
    def execute_trajectory(
        self, times, joint_traj, tolerance_deg=2.0, timeout=3.0
    ):
        # An interrupted/failed move must never leave a reusable stale endpoint.
        self._last_arm_command_deg = None
        if len(times) != len(joint_traj):
            raise ValueError("Each timestamp must have one joint command")
        if not joint_traj:
            return

        start_time = time.perf_counter()

        for target_time, joint_action in zip(times, joint_traj):
            delay = start_time + float(target_time) - time.perf_counter()

            if delay > 0:
                time.sleep(delay)

            self.arm.send_action(joint_action)

        # Wait for the arm to reach the final target.
        final_target = joint_traj[-1]
        settle_start = time.perf_counter()
        deadline = settle_start + timeout
        next_report = settle_start
        recent_positions = deque()

        while True:
            observation = self.arm.get_observation()
            now = time.perf_counter()
            actual = {
                name: float(observation[f"{name}.pos"])
                for name in self.arm_joint_names
            }
            signed_errors = {
                name: float(final_target[f"{name}.pos"]) - actual[name]
                for name in self.arm_joint_names
            }
            errors = {name: abs(error) for name, error in signed_errors.items()}
            # Retain roughly the last half-second, including the sample just
            # before its boundary. Range captures oscillation as well as drift.
            recent_positions.append((now, actual))
            while len(recent_positions) > 2 and recent_positions[1][0] < now - 0.5:
                recent_positions.popleft()

            if max(errors.values()) <= tolerance_deg:
                self._last_arm_command_deg = {
                    name: float(final_target[f"{name}.pos"])
                    for name in self.arm_joint_names
                }
                return

            failing = [name for name, error in errors.items() if error > tolerance_deg]
            if now >= next_report:
                details = "; ".join(
                    f"{name}: target={float(final_target[f'{name}.pos']):+.2f}°, "
                    f"actual={actual[name]:+.2f}°, error={signed_errors[name]:+.2f}°"
                    for name in failing
                )
                print(f"[SO101 settling {now - settle_start:.2f}/{timeout:.2f}s] {details}")
                next_report = now + 0.5

            if now >= deadline:
                window = now - recent_positions[0][0]
                details = [
                    f"Final target not reached after {now - settle_start:.2f}s "
                    f"of settling (tolerance={tolerance_deg:.2f}°).",
                    "Angles below are LeRobot servo degrees; signed error = target - actual.",
                ]
                for name in self.arm_joint_names:
                    values = [positions[name] for _, positions in recent_positions]
                    details.append(
                        f"  {name}: target={float(final_target[f'{name}.pos']):+.3f}°, "
                        f"actual={actual[name]:+.3f}°, error={signed_errors[name]:+.3f}°, "
                        f"recent_range={max(values) - min(values):.3f}° "
                        f"over {window:.2f}s"
                        + (" -- OUTSIDE tolerance" if name in failing else "")
                    )
                # Best-effort reads only, after the timeout. They must neither
                # alter torque/gains nor hide the original tracking failure.
                details.extend(self._read_timeout_motor_diagnostics(failing))
                raise TimeoutError("\n".join(details))

            time.sleep(0.02)

    def _read_timeout_motor_diagnostics(self, joint_names):
        """Read the failed joints' registers without writing or reconfiguring."""
        registers = [
            ("Goal_Position_deg", "Goal_Position", True),
            ("Goal_Position_raw", "Goal_Position", False),
        ] + [
            (name, name, False) for name in (
                "Min_Position_Limit", "Max_Position_Limit", "Operating_Mode",
                "P_Coefficient", "I_Coefficient", "D_Coefficient",
                "CW_Dead_Zone", "CCW_Dead_Zone",
                "Torque_Enable", "Torque_Limit", "Max_Torque_Limit",
                "Present_Velocity", "Present_Load", "Present_Current",
                "Present_Voltage", "Present_Temperature", "Status", "Moving",
            )
        ]
        readings = {name: [] for name in joint_names}
        unavailable = []
        for label, register, normalize in registers:
            try:
                values = self.arm.bus.sync_read(
                    register, motors=joint_names, normalize=normalize, num_retry=0,
                )
                for name in joint_names:
                    readings[name].append(f"{label}={float(values[name]):.3f}")
            except Exception as exc:
                unavailable.append(f"  {label}: unavailable ({type(exc).__name__}: {exc})")
        return [
            "Read-only motor snapshot after timeout (Goal_Position_deg in degrees; "
            "other registers in raw units):",
            *(f"  {name}: {', '.join(values)}" for name, values in readings.items()),
            *unavailable,
        ]

    def go_to_cartesian_pose(self, positions, orientations=None, *, fallback_positions=None,
                             travel_z=None, continuous_range=None):
        """Plan the full route before moving; optionally try an alternate route.

        fallback_positions is a complete route in FRIDA coordinates. It is
        tried only after a planning error, never after execution starts.
        travel_z requests a lift/traverse/descent to one destination. It cannot
        be combined with a fallback route that might bypass the clearance.
        """
        if travel_z is not None and fallback_positions is not None:
            raise ValueError("A clearance transfer cannot use fallback_positions")
        if continuous_range is not None and (travel_z is not None or fallback_positions is not None):
            raise ValueError("A continuous stroke cannot use travel_z or fallback_positions")
        previous_command = getattr(self, "_last_arm_command_deg", None)
        self._last_arm_command_deg = None
        observation = self.arm.get_observation()
        # LeRobot reads and commands the gripper in the same 0-100 units.
        # Capture once for this trajectory and reuse the value unchanged.
        gripper_position = float(observation["gripper.pos"])

        current_joints_deg = {
            name: float(observation[f"{name}.pos"])
            for name in self.arm_joint_names
        }
        
        # Nominal estimate for collision geometry: closed/open map to the
        # URDF limits. This is not a calibrated servo-to-URDF zero alignment.
        limits = self.motion_planner.scene.getJointInfo("gripper").limits
        lower, upper = limits.min_position[0], limits.max_position[0]
        current_joints_deg["gripper"] = float(np.rad2deg(
            lower + (gripper_position / 100.0) * (upper - lower)
        ))

        robot_positions = self.frida_to_robot(
            np.atleast_2d(np.asarray(positions, dtype=float))
        )

        # Only continue a successfully completed command while feedback remains
        # nearby. After manual repositioning or a failure, plan from measurement.
        planning_options = {}
        if continuous_range is not None:
            planning_options["continuous_range"] = continuous_range
        if travel_z is not None:
            # FRIDA and robot-base coordinates have the same vertical axis.
            planning_options["travel_z"] = travel_z
        if previous_command is not None and all(
            np.isfinite(current_joints_deg[name])
            and abs(current_joints_deg[name] - previous_command[name]) <= 2.0
            for name in self.arm_joint_names
        ):
            planning_options["previous_command_deg"] = previous_command
        try:
            trajectory = self.motion_planner.plan_xyz_path(
                robot_positions, current_joints_deg, **planning_options
            )
        except (RuntimeError, ValueError) as direct_error:
            if fallback_positions is None:
                raise MotionPlanningError(
                    f"Cartesian route planning failed: {direct_error}"
                ) from direct_error
            print("Direct route planning failed; trying the fallback route.", flush=True)
            fallback_robot_positions = self.frida_to_robot(
                np.atleast_2d(np.asarray(fallback_positions, dtype=float))
            )
            # Reuse the same measured start. Both legs must pass all planner
            # checks before any trajectory commands are sent.
            try:
                trajectory = self.motion_planner.plan_xyz_path(
                    fallback_robot_positions, current_joints_deg, **planning_options
                )
            except (RuntimeError, ValueError) as fallback_error:
                raise MotionPlanningError(
                    "Cartesian direct and fallback routes failed planning: "
                    f"direct: {direct_error}; fallback: {fallback_error}"
                ) from fallback_error
        if trajectory is None:
            self._last_arm_command_deg = planning_options.get("previous_command_deg")
            return True

        commands = []

        for q_model in trajectory.positions:
            # Invert the same startup calibration used by plan_xyz_path:
            # model_deg = observation_deg + offset_deg.
            action = {
                f"{name}.pos": (
                    float(np.rad2deg(angle))
                    - self.motion_planner.joint_offsets_deg.get(name, 0.0)
                )
                for name, angle in zip(trajectory.joint_names, q_model)
            }
            action["gripper.pos"] = gripper_position
            commands.append(action)

        self.execute_trajectory(trajectory.times, commands)
            
    def get_cartesian_pose(self):
        """Return the measured brush-tip [x, y, z] in FRIDA coordinates, meters."""
        observation = self.arm.get_observation()
        planner = self.motion_planner
        q = planner.scene.getCurrentJointPositions().copy()
        for name in planner.joint_names:
            angle = float(observation[f"{name}.pos"])
            if not np.isfinite(angle):
                raise ValueError(f"Invalid angle for {name}")
            index = planner.scene.getJointPositionIndices([name])[0]
            q[index] = np.deg2rad(angle + planner.joint_offsets_deg[name])

        # The brush is fixed to the gripper body; jaw opening does not affect FK.
        pose = planner.scene.forwardKinematics(q, planner.tip_frame, planner.base_frame)
        # Inverse of frida_to_robot: robot [x, y, z] -> FRIDA [-y, x, z].
        return [-float(pose[1, 3]), float(pose[0, 3]), float(pose[2, 3])]
