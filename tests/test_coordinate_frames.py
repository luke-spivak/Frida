import numpy as np
from robot import SO101
from scipy.spatial.transform import Rotation as R


PERPENDICULAR_QUATERNION = np.array([1.77622069e-04,   9.23865441e-01,  -3.82717408e-01,  -1.73978366e-05])

if __name__ == "__main__":
    robot = SO101(debug=True)
    robot.good_morning_robot()

    obs = robot.arm.get_observation()
    q = np.array([float(obs[f"{n}.pos"]) for n in robot.arm_joint_names])
    T = robot.kinematics.forward_kinematics(q)

    print("Tool X-axis in base frame:", T[:3, 0])
    print("Tool Y-axis in base frame:", T[:3, 1])
    print("Tool Z-axis in base frame:", T[:3, 2])
    robot.good_night_robot()
    # try:
    #     # position = [0.0838, 0.1862, 0.0711]
        
    #     # Keep attitude the same
    #     # Get position in arm frame
    #     ee = robot.get_cartesian_pose()
        
    #     # Transform position to frida frame
    #     position = [-float(ee["ee.y"]), float(ee["ee.x"]), float(ee["ee.z"])] #  x = -y, y = x
    #     print("Neutral position:", position)
    #     position[1] += 0.20
        
    #     rot_vec = [float(ee['ee.wx']), float(ee['ee.wy']), float(ee['ee.wz'])]
    #     q = R.from_rotvec(rot_vec).as_quat()
    #     print("Brush Q", q)
        

    #     # robot.go_to_cartesian_pose(position, [None])

    # finally:
    #     robot.good_night_robot()