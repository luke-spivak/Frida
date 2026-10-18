import numpy as np

from roboplan.core import (
    Scene,
    CartesianPath,
    JointConfiguration,
    loadUrdfSceneDescription,
    loadJointLimitsConfig,
    hasCollisionsAlongPath,
)
from roboplan.cartesian_planning import (
    CartesianPathPlanner,
    CartesianPlannerOptions,
)
from roboplan.core import CartesianConfiguration, JointPath
from roboplan.optimal_ik import Oink, FrameTask, FrameTaskOptions, PositionLimit
from roboplan.toppra import (
    PathParameterizerTOPPRA,
    TOPPRAOptions,
    SplineFittingMode,
)

from pathlib import Path
import pinocchio as pin
from pinocchio.visualize import ViserVisualizer


class SO101MotionPlanner:

    def __init__(
        self, 
        urdf_path="SO101/so101_brush_tip_nominal.urdf",
        srdf_path="SO101/so101_arm.srdf",
        joint_limits_path="SO101/so101_config.yaml",
        show_viewer=False
    ):
        self.urdf = Path(urdf_path).resolve()
        self.package_dirs = [str(self.urdf.parent)]
        self.group_name = "manipulator"
        self.base_frame = "base_link" # TODO: Change to table frame?
        self.tip_frame = "brush_tip"
        
        self.scene = Scene(
            "painting",
            loadUrdfSceneDescription(
                str(self.urdf), self.package_dirs
            ),
        )
        self.scene.importSrdf(Path(srdf_path).read_text())
        self.scene.importJointLimitsFromConfig(
            loadJointLimitsConfig(Path(joint_limits_path))
        )

        self.joint_names = list(
            self.scene.getJointGroupInfo(self.group_name).joint_names
        )
        
        self.ik = Oink(self.scene, self.group_name)
        self.ik_limits = [PositionLimit(self.ik)]
        self.arm_indices = np.asarray(
            self.scene.getJointPositionIndices(self.joint_names),
            dtype=int,
        )
        self.parameterizer = PathParameterizerTOPPRA(
            self.scene, self.group_name
        )
        
        self.viewer = None
        if show_viewer:
            self._create_viewer()

    def _create_viewer(self):
        import pinocchio as pin
        from pinocchio.visualize import ViserVisualizer

        self.viewer_model = pin.buildModelFromUrdf(str(self.urdf))
        collision = pin.buildGeomFromUrdf(
            self.viewer_model,
            str(self.urdf),
            pin.GeometryType.COLLISION,
            package_dirs=self.package_dirs,
        )
        visual = pin.buildGeomFromUrdf(
            self.viewer_model,
            str(self.urdf),
            pin.GeometryType.VISUAL,
            package_dirs=self.package_dirs,
        )

        self.viewer = ViserVisualizer(
            self.viewer_model, collision, visual
        )
        self.viewer.initViewer(
            open=True, loadModel=True, host="localhost", port=8000
        )
        self.viewer.display(pin.neutral(self.viewer_model))
        
    def _solve_xyz(self, xyz, seed):
        """Solve one XYZ target. Full-model seed/result, in radians."""
        q = seed.copy()

        target = self.scene.forwardKinematics(
            q, self.tip_frame, self.base_frame
        ).copy()
        target[:3, 3] = xyz

        task = FrameTask(
            self.ik,
            self.scene,
            CartesianConfiguration(
                self.base_frame, self.tip_frame, target
            ),
            FrameTaskOptions(
                position_cost=1.0,
                orientation_cost=0.0,
                task_gain=0.5,
                lm_damping=0.001,
            ),
        )

        delta = np.zeros(self.ik.num_variables, dtype=np.float64)

        for _ in range(300):
            self.scene.setJointPositions(q)
            achieved = self.scene.forwardKinematics(
                q, self.tip_frame, self.base_frame
            )
            error = np.linalg.norm(achieved[:3, 3] - xyz)

            if error <= 0.001:
                return q

            delta.fill(0.0)
            self.ik.solveIk(
                self.scene, [task], self.ik_limits, [], delta, 1e-6
            )

            if not np.all(np.isfinite(delta)):
                raise RuntimeError("IK produced non-finite joint increments")

            # Limit numerical IK steps; this is not execution timing.
            largest_step = np.max(np.abs(delta))
            if largest_step > np.deg2rad(2):
                delta *= np.deg2rad(2) / largest_step

            q[self.ik.q_indices] += delta

        raise RuntimeError(f"Waypoint IK failed for XYZ {xyz}")
    
    def plan_xyz_path(self, positions, current_joints_deg):
        """
        positions: target XYZ points in robot base coordinates, meters.
        current_joints_deg: {URDF joint name: angle in degrees},
                            including the gripper's MODEL angle.

        Returns a RoboPlan JointTrajectory. Does not move the arm.
        """
        points = np.asarray(positions, dtype=float)
        if points.shape == (3,):
            points = points[None, :]
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or len(points) == 0
            or not np.all(np.isfinite(points))
        ):
            raise ValueError("Expected finite XYZ points with shape (N, 3)")

        # Populate the complete model state by joint name.
        q_full = self.scene.getCurrentJointPositions().copy()
        for name in self.scene.getJointNames():
            angle = float(current_joints_deg[name])
            if not np.isfinite(angle):
                raise ValueError(f"Invalid angle for {name}")
            indices = self.scene.getJointPositionIndices([name])
            if len(indices) != 1:
                raise ValueError(f"Expected a bounded revolute joint: {name}")
            q_full[indices[0]] = np.deg2rad(angle)

        self.scene.setJointPositions(q_full)

        q = q_full.copy()
        joint_waypoints = [q[self.arm_indices].copy()]

        for i, xyz in enumerate(points):
            q_next = self._solve_xyz(xyz, q)

            # Check the connecting joint-space segment, including its endpoints.
            if hasCollisionsAlongPath(
                self.scene, q, q_next, max_step_size=0.01
            ):
                raise RuntimeError(
                    f"Joint-space segment to waypoint {i} collides"
                )

            # Avoid duplicate waypoints in the time parameterizer.
            if np.max(np.abs(q_next - q)) > 1e-8:
                joint_waypoints.append(q_next[self.arm_indices].copy())

            q = q_next

        if len(joint_waypoints) < 2:
            return None
        
        path = JointPath()
        path.joint_names = self.joint_names
        path.positions = joint_waypoints

        # Restore the measured full state, including the fixed gripper.
        self.scene.setJointPositions(q_full)

        trajectory = self.parameterizer.generate(
            path,
            TOPPRAOptions(
                dt=0.02,
                mode=SplineFittingMode.Cubic,
                velocity_scale=0.1,
                acceleration_scale=0.1,
            ),
        )
        if len(trajectory.times) == 0:
            raise RuntimeError("Joint trajectory parameterization failed")

        # Validate the returned motion before allowing execution.
        output_indices = self.scene.getJointPositionIndices(
            trajectory.joint_names
        )
        previous = q_full.copy()
        for q in trajectory.positions:
            current = q_full.copy()
            current[output_indices] = q
            if hasCollisionsAlongPath(
                self.scene, previous, current, max_step_size=0.01
            ):
                raise RuntimeError("Planned trajectory intersects an obstacle")
            previous = current

        return trajectory