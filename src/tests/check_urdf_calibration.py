#!/usr/bin/env python3
"""
Check whether so101_new_calib.urdf matches the physical SO-101 calibration.

Usage:
    python check_urdf_calibration.py

What it does:
1. Connects to the follower arm.
2. Reads current joints and computes FK → prints predicted EE pose.
3. Moves the arm to several canonical joint configurations.
4. For each pose prints the FK-predicted (x,y,z) so you can measure it
   with a ruler / tape / phone measure app relative to the base.
5. Performs a small IK→FK round-trip test around the current pose.
"""

import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower
from lerobot.model.kinematics import RobotKinematics


# ============== USER SETTINGS ==============
PORT = "/dev/tty.usbmodem5B8E1155331"         # <-- your port
URDF_PATH = "./SO101/so101_new_calib.urdf"
TARGET_FRAME = "gripper_frame_link"
USE_DEGREES = True
ROBOT_ID = "my_arm"

# Joint order expected by the kinematics object (no gripper)
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
# ===========================================


def get_joint_vector(obs, joint_names):
    """Extract joint positions in the order expected by kinematics (degrees)."""
    return np.array([float(obs[f"{name}.pos"]) for name in joint_names])


def print_pose(label, T):
    pos = T[:3, 3]
    rpy = R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
    print(f"\n--- {label} ---")
    print(f"  Position (m):  x={pos[0]:+.4f}  y={pos[1]:+.4f}  z={pos[2]:+.4f}")
    print(f"  RPY (deg):     r={rpy[0]:+.1f}  p={rpy[1]:+.1f}  y={rpy[2]:+.1f}")
    return pos


def move_to_joints(arm, target_joints_deg, joint_names, timeout=12.0, tol=3.0):
    """Send absolute joint command and wait until close."""
    action = {f"{name}.pos": float(q) for name, q in zip(joint_names, target_joints_deg)}
    # Keep gripper wherever it is
    obs = arm.get_observation()
    action["gripper.pos"] = float(obs["gripper.pos"])

    arm.send_action(action)

    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        obs = arm.get_observation()
        q = get_joint_vector(obs, joint_names)
        if np.all(np.abs(q - target_joints_deg) < tol):
            return True
        time.sleep(0.05)
    return False


def main():
    print("Connecting to SO-101...")
    config = SO101FollowerConfig(
        port=PORT,
        id=ROBOT_ID,
        use_degrees=USE_DEGREES,
        max_relative_target=None,
    )
    arm = SO101Follower(config=config)
    arm.connect()

    kin = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name=TARGET_FRAME,
        joint_names=JOINT_NAMES,
    )

    # ------------------------------------------------------------------
    # 1. Current pose
    # ------------------------------------------------------------------
    obs = arm.get_observation()
    q_curr = get_joint_vector(obs, JOINT_NAMES)
    print("\nCurrent joint angles (deg):")
    for name, q in zip(JOINT_NAMES, q_curr):
        print(f"  {name:15s}: {q:+7.2f}")

    T_curr = kin.forward_kinematics(q_curr)
    pos_curr = print_pose("Current pose (FK from URDF)", T_curr)

    input("\n>>> Measure the real gripper position relative to the base now.\n"
          "    Compare with the numbers above. Press Enter to continue...")

    # ------------------------------------------------------------------
    # 2. Canonical poses (edit the joint values if they are unsafe for you)
    # ------------------------------------------------------------------
    # These are approximate "new-calib" middle-range poses.
    # Adjust if any of them cause self-collision on your particular print.
    test_poses = {
        "Near-zero (middle of ranges)": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "Slightly raised":              np.array([0.0, -30.0, 40.0, 20.0, 0.0]),
        "Forward reach":                np.array([0.0, -45.0, 70.0, 30.0, 0.0]),
        "Side reach (pan +45)":         np.array([45.0, -30.0, 50.0, 20.0, 0.0]),
    }

    measured = {}
    for name, q_des in test_poses.items():
        print(f"\nMoving to: {name}")
        print("  target joints:", np.round(q_des, 1))
        ok = move_to_joints(arm, q_des, JOINT_NAMES)
        if not ok:
            print("  WARNING: did not reach target within tolerance")
        time.sleep(0.5)

        obs = arm.get_observation()
        q = get_joint_vector(obs, JOINT_NAMES)
        T = kin.forward_kinematics(q)
        pos = print_pose(name, T)
        measured[name] = pos

        input("  >>> Measure real (x,y,z) of the gripper now, then press Enter...")

    # ------------------------------------------------------------------
    # 3. Round-trip IK → FK test around current pose
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("IK → FK round-trip test (should be < 5-10 mm if model is good)")
    print("="*60)

    obs = arm.get_observation()
    q0 = get_joint_vector(obs, JOINT_NAMES)
    T0 = kin.forward_kinematics(q0)

    offsets = [
        ("+2 cm X", np.array([0.02, 0.0, 0.0])),
        ("+2 cm Y", np.array([0.0, 0.02, 0.0])),
        ("+2 cm Z", np.array([0.0, 0.0, 0.02])),
        ("-2 cm Z", np.array([0.0, 0.0, -0.02])),
    ]

    for label, dpos in offsets:
        T_tgt = T0.copy()
        T_tgt[:3, 3] += dpos

        q_ik = kin.inverse_kinematics(
            q0,
            T_tgt,
            position_weight=1.0,
            orientation_weight=0.0,   # position-only for this test
            max_iters=20,
        )
        T_ach = kin.forward_kinematics(q_ik)
        err_mm = np.linalg.norm(T_tgt[:3, 3] - T_ach[:3, 3]) * 1000.0
        print(f"  {label:12s}  position error = {err_mm:6.2f} mm")

    # ------------------------------------------------------------------
    # 4. Summary guidance
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print("""
• If FK positions are consistently off by a nearly constant offset
  (e.g. always +3 cm in Z) → zero-point / calibration convention mismatch.
  Re-calibrate carefully with every joint at the middle of its travel first.

• If error grows with arm extension → link lengths in the URDF do not
  match your physical print. Measure upper-arm / forearm and edit the URDF.

• If round-trip errors above are already > 15-20 mm even for 2 cm moves
  → URDF frame name, joint order, or axis directions are wrong.

• Good match: measured vs FK within ~5-10 mm (plus normal backlash).
""")

    arm.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()