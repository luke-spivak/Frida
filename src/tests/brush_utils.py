from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower
from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower.robot_kinematic_processor import (
    ForwardKinematicsJointsToEEObservation
)
from lerobot.processor import (
    RobotProcessorPipeline,
    transition_to_observation,
    observation_to_transition,
    
)
from lerobot.lerobot_types import RobotAction, RobotObservation
from scipy.spatial.transform import Rotation as R
        
        
def get_cartesian_pose(arm: SO101Follower) -> tuple[float, float, float]:
    motor_names = list(arm.bus.motors.keys())
    
    kinematics = RobotKinematics(
        urdf_path="./SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=motor_names,
    )
    
    # Get current state of joint angles
    
    joints_to_follower_ee = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[ForwardKinematicsJointsToEEObservation(
            kinematics=kinematics,
            motor_names=motor_names,
            )
        ],
        to_transition=observation_to_transition,
        to_output=transition_to_observation
    )
    observation = arm.get_observation()
    ee = joints_to_follower_ee(observation)
    position = float(ee["ee.x"]), float(ee["ee.y"]), float(ee["ee.z"])
    rotvec = [float(ee[k]) for k in ("ee.wx", "ee.wy", "ee.wz")]
    return float(ee["ee.x"]), float(ee["ee.y"]), float(ee["ee.z"]), rotvec


if __name__ == "__main__":
    # Config for arm
    config = SO101FollowerConfig(
        port="/dev/tty.usbmodem5B8E1155331",
        id="my_arm",
        use_degrees=True,
    )
    arm = SO101Follower(config=config)
    arm.connect()

    try:
        input("Support the arm, then press Enter to disable torque.")
        arm.bus.disable_torque()

        while True:
            input("Position the arm, then press Enter to read XYZ: ")
            x, y, z, rotvec = get_cartesian_pose(arm)
            print(f"Position (meters): ({x:.4f}, {y:.4f}, {z:.4f})")

    except KeyboardInterrupt:
        pass
    finally:
        arm.disconnect()
# old
# Water Position (meters): (0.3602, 0.2140, 0.0275)
# Towel Position (meters): (0.2489, 0.2127, 0.0294)
# Canvas Position (meters): (
# , -0.0024, 0.0326)
# (Placeholder) Paint Position (meters): (0.3865, 0.1533, 0.0206) 
# INIT_TABLE_Z (0.0238) m
# MAX_TABLE_Z (0.0226) m

# new
# Rag Position (meters): (0.1435, 0.2355, 0.0370)
# Water Position (meters): (0.2711, 0.1991, 0.0308)
# Paint Position (meters): (0.3347, 0.1540, 0.0243)