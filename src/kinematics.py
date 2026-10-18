from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.import_utils import require_package

_placo_runtime_error: ImportError | None = None

if TYPE_CHECKING:
    import placo
else:
    try:
        import placo  # type: ignore[import-not-found]
    except ImportError as _placo_import_err:
        placo = None
        _placo_runtime_error = _placo_import_err


def _raise_if_placo_unusable() -> None:
    if placo is None and _placo_runtime_error is not None:
        raise ImportError(
            f"placo is installed but failed to import: {_placo_runtime_error!s}"
        ) from _placo_runtime_error


class RobotKinematics:
    """Robot kinematics using placo library for forward and inverse kinematics."""

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str,
        joint_names: list[str] | None,
        brush_axis_local,
        brush_direction_world=(0.0, 0.0, -1.0),
    ):
        """
        Initialize placo-based kinematics solver.

        Args:
            urdf_path (str): Path to the robot URDF file
            target_frame_name (str): Name of the end-effector frame in the URDF
            joint_names (list[str] | None): List of joint names to use for the kinematics solver
        """
        require_package("placo", extra="placo-dep")
        _raise_if_placo_unusable()

        self.robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)  # Fix the base
        self.solver.enable_joint_limits(True)

        self.target_frame_name = target_frame_name

        # Set joint names
        self.joint_names = list(self.robot.joint_names()) if joint_names is None else joint_names
        
        self.brush_axis_local = self._unit_vector(brush_axis_local)
        self.brush_direction_world = self._unit_vector(
            brush_direction_world
        )
        
        # Prevent IK from moving joints outside our controlled arm joints.
        # These remain at their model positions.
        for name in self.robot.joint_names():
            if name not in self.joint_names:
                self.solver.mask_dof(name)

        self.position_task = self.solver.add_position_task(
            target_frame_name, np.zeros(3)
        )
        self.axis_task = self.solver.add_axisalign_task(
            target_frame_name,
            self.brush_axis_local,
            self.brush_direction_world, 
        )
        
    @staticmethod
    def _unit_vector(value):
        vector = np.asarray(value, dtype=float)
        if (
            vector.shape != (3,)
            or not np.all(np.isfinite(vector))
            or np.linalg.norm(vector) < 1e-12
        ):
            raise ValueError("Expected a finite, nonzero 3D axis")
        return vector / np.linalg.norm(vector)
    
    def _set_joints(self, joint_pos_deg):
        q = np.asarray(joint_pos_deg, dtype=float)
        n = len(self.joint_names)

        if q.ndim != 1 or len(q) < n or not np.all(np.isfinite(q)):
            raise ValueError("Invalid joint configuration")

        # Convert degrees to radians and update joint positions in placo robot
        for name, angle in zip(self.joint_names, np.deg2rad(q[:n])):
            self.robot.set_joint(name, float(angle))

        self.robot.update_kinematics()
        

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for given joint configuration given the target frame name in the constructor.

        Args:
            joint_pos_deg: Joint positions in degrees (numpy array)

        Returns:
            4x4 transformation matrix of the end-effector pose
        """
        # Update joint positions in placo robot
        self._set_joints(joint_pos_deg)
        # Get the transformation matrix
        return self.robot.get_T_world_frame(
            self.target_frame_name
        ).copy()
        
    def brush_tilt_deg(self, pose):
        """Angle between the brush shaft and its desired direction."""
        actual_axis = pose[:3, :3] @ self.brush_axis_local
        cosine = np.dot(actual_axis, self.brush_direction_world)
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 1.0,
        max_iters: int = 50,
        position_tolerance=0.001,
        tilt_tolerance_deg=2.0,
    ) -> np.ndarray:
        """
        Compute inverse kinematics using placo solver.
        
        desired_ee_pose rotation is intentionally ignored.
        orientation_weight now weights brush-axis alignment.

        Args:
            current_joint_pos: Current joint positions in degrees (used as initial guess)
            desired_ee_pose: Target end-effector pose as a 4x4 transformation matrix
            position_weight: Weight for position constraint in IK
            orientation_weight: Weight for orientation constraint in IK, set to 0.0 to only constrain position
            max_iters: Number of placo Newton steps to run.

        Returns:
            Joint positions in degrees that achieve the desired end-effector pose
        """
        target = np.asarray(desired_ee_pose, dtype=float)
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            raise ValueError("Expected a finite 4×4 target pose")

        self._set_joints(current_joint_pos)

        self.position_task.target_world = target[:3, 3]
        self.position_task.configure(
            "brush_position", "hard", # position_weight
        )
        self.axis_task.configure(
            "brush_axis", "soft", orientation_weight
        )

        # Solve IK.
        for iteration in range(max_iters + 1):
            pose = self.robot.get_T_world_frame(
                self.target_frame_name
            ).copy()

            position_error = np.linalg.norm(
                pose[:3, 3] - target[:3, 3]
            )
            tilt_error = self.brush_tilt_deg(pose)
            
            if (
                position_error <= position_tolerance
                # and tilt_error <= tilt_tolerance_deg
            ):
                result = np.asarray(
                    current_joint_pos, dtype=float
                ).copy()

                # Convert back to degrees
                result[:len(self.joint_names)] = np.rad2deg([
                    self.robot.get_joint(name)
                    for name in self.joint_names
                ])
                return result

            if iteration < max_iters:
                try:
                    self.solver.solve(True)
                    self.robot.update_kinematics()
                except RuntimeError as exc:
                    if "Infeasible QP" not in str(exc):
                        raise  # Don't hide unrelated errors.
                    
                    return None
                
                

        raise ValueError(
            f"IK failed: position error={position_error * 1000:.3f} mm, "
            f"brush tilt={tilt_error:.2f} deg"
        )
