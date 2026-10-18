import numpy as np
from robot import SO101
from scipy.spatial.transform import Rotation as R


if __name__ == "__main__":
    robot = SO101(debug=True)
    robot.good_morning_robot()

    try:
        input("Support the arm, then press Enter to disable torque.")
        robot.arm.bus.disable_torque()
    
        while True:
            label = input(
                "Place tip on the marked point, hold still, "
                "then enter pose label (or q to quit): "
            )
            if label.lower() == "q":
                break

            obs = robot.arm.get_observation()
            joints = np.array([
                float(obs[f"{name}.pos"])
                for name in robot.arm_joint_names
            ])
            T = robot.kinematics.forward_kinematics(joints).copy()

            frida_xyz = np.array([-T[1, 3], T[0, 3], T[2, 3]])

            print({
                "label": label,
                "joints_deg": dict(zip(robot.arm_joint_names, joints.tolist())),
                "frida_xyz_m": (frida_xyz).tolist(),
                "tip_rotation_matrix": T[:3, :3].tolist(),
            })
            robot.get_cartesian_pose()

    finally:
        robot.good_night_robot()
        # New 
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