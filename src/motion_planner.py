import numpy as np
import time
import json
import tempfile
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

from roboplan.core import (
    Scene,
    JointTrajectory,
    loadUrdfSceneDescription,
    loadJointLimitsConfig,
    hasCollisionsAlongPath,
)
from roboplan.core import CartesianConfiguration
from roboplan.optimal_ik import (
    Oink,
    FrameTask,
    FrameTaskOptions,
    PositionLimit,
)

from pathlib import Path
import pinocchio as pin
from pinocchio.visualize import ViserVisualizer


DEFAULT_JOINT_OFFSETS_PATH = Path(__file__).resolve().with_name("so101_offsets.json")
DEFAULT_LEROBOT_CALIBRATION_PATH = (
    Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/my_arm.json"
)
# LeRobot's STS3215 DEGREES conversion uses resolution - 1, not 4096.
SO101_ENCODER_MAX = 4095


def load_joint_offsets(path, joint_names):
    """Load model-minus-observation degrees; never use the saved observation."""
    path = Path(path).expanduser().resolve()
    try:
        saved = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Joint offsets file not found: {path}. Save offsets with "
            "preview_so101_offsets.py before starting the robot."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid joint offsets JSON in {path}: {exc}") from exc

    offsets = saved.get("joint_offsets_deg", saved) if isinstance(saved, dict) else saved
    if not isinstance(offsets, dict):
        raise ValueError(f"Expected a joint_offsets_deg mapping in {path}")
    missing = set(joint_names) - offsets.keys()
    unknown = offsets.keys() - set(joint_names)
    if missing or unknown:
        raise ValueError(
            f"Invalid joint offsets in {path}: missing joints={sorted(missing)}, "
            f"unknown joints={sorted(unknown)}"
        )
    result = {}
    for name in joint_names:
        value = offsets[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"Offset for {name} in {path} must be a finite number of degrees")
        result[name] = float(value)
    return result


def load_calibrated_joint_limits(path, joint_offsets_deg):
    """Convert saved LeRobot encoder ranges to model-coordinate degree bounds."""
    path = Path(path).expanduser().resolve()
    try:
        calibration = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"LeRobot calibration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid LeRobot calibration JSON in {path}: {exc}") from exc
    if not isinstance(calibration, dict):
        raise ValueError(f"Expected a LeRobot joint calibration mapping in {path}")

    limits_deg = {}
    for name, offset in joint_offsets_deg.items():
        entry = calibration.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"Missing or invalid calibration for {name} in {path}")
        lower, upper = entry.get("range_min"), entry.get("range_max")
        if (
            type(lower) is not int or type(upper) is not int
            or not 0 <= lower < upper <= SO101_ENCODER_MAX
        ):
            raise ValueError(
                f"Invalid encoder range for {name} in {path}: require integer "
                f"0 <= range_min < range_max <= {SO101_ENCODER_MAX}"
            )
        # Present_Position and these endpoints already include the servo homing
        # offset. Match LeRobot's midpoint-based DEGREES normalization exactly;
        # do not apply homing_offset a second time. The gripper is not in this map.
        midpoint = (lower + upper) / 2.0
        limits_deg[name] = (
            (lower - midpoint) * 360.0 / SO101_ENCODER_MAX + offset,
            (upper - midpoint) * 360.0 / SO101_ENCODER_MAX + offset,
        )
    return limits_deg


class SO101MotionPlanner:

    def __init__(
        self,
        urdf_path="SO101/so101_brush_tip_nominal.urdf",
        srdf_path="SO101/so101_arm.srdf",
        joint_limits_path="SO101/so101_config.yaml",
        show_viewer=False,
        joint_offsets_path=None,
        lerobot_calibration_path=None,
        speed_scale=4.0,
    ):
        if not np.isfinite(speed_scale) or speed_scale <= 0:
            raise ValueError("speed_scale must be positive and finite")
        self.speed_scale = float(speed_scale)
        self.urdf = Path(urdf_path).resolve()
        self.package_dirs = [str(self.urdf.parent)]
        self.group_name = "manipulator"
        self.base_frame = "base_link" # TODO: Change to table frame?
        self.tip_frame = "brush_tip"
        self.shaft_frame = "brush_exposed_visual"
        self.brush_direction = np.array([0.0, 0.0, -1.0])
        self.brush_tilt_tolerance_deg = 10.0

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
        # Two points on the rigid shaft define its direction without fixing spin.
        tip_to_shaft = self.scene.forwardKinematics(
            self.scene.getCurrentJointPositions(), self.shaft_frame, self.tip_frame,
        )[:3, 3]
        self.brush_axis_length = float(np.linalg.norm(tip_to_shaft))
        if self.brush_axis_length < 1e-6:
            raise ValueError("Brush shaft frame must be distinct from the tip")
        self.brush_axis_local = -tip_to_shaft / self.brush_axis_length

        self.joint_names = list(
            self.scene.getJointGroupInfo(self.group_name).joint_names
        )
        self.joint_offsets_path = (
            DEFAULT_JOINT_OFFSETS_PATH if joint_offsets_path is None
            else Path(joint_offsets_path).expanduser().resolve()
        )
        # Capture one calibration for the entire planning/execution session.
        self.joint_offsets_deg = load_joint_offsets(self.joint_offsets_path, self.joint_names)
        print(f"Loaded joint offsets from {self.joint_offsets_path}: {self.joint_offsets_deg}", flush=True)
        self.lerobot_calibration_path = Path(
            DEFAULT_LEROBOT_CALIBRATION_PATH if lerobot_calibration_path is None
            else lerobot_calibration_path
        ).expanduser().resolve()
        self.joint_limits_deg = load_calibrated_joint_limits(
            self.lerobot_calibration_path, self.joint_offsets_deg,
        )
        # Use RoboPlan's importer so the scene's model, validity checks and RNG
        # all receive the same bounds BEFORE IK and trajectory objects are made.
        # Its Python YAML loader only accepts a file; JSON is valid YAML.
        overrides = {"joint_limits": {
            name: {"min_position": [float(np.deg2rad(lower))],
                   "max_position": [float(np.deg2rad(upper))]}
            for name, (lower, upper) in self.joint_limits_deg.items()
        }}
        with tempfile.TemporaryDirectory(prefix="so101-joint-limits-") as directory:
            path = Path(directory) / "limits.json"
            path.write_text(json.dumps(overrides))
            self.scene.importJointLimitsFromConfig(loadJointLimitsConfig(path))
        print(
            f"Loaded calibrated model joint limits (deg) from "
            f"{self.lerobot_calibration_path}: {self.joint_limits_deg}", flush=True,
        )

        self.ik = Oink(self.scene, self.group_name)
        self.ik_limits = [PositionLimit(self.ik)]

        # The outer loop retries both failed IK and blocked connections.
        self.ik_max_attempts = 10
        self.ik_max_iters = 300
        self.ik_attempt_timeout = 15.0
        # Resolve the geometry more accurately than the checked path tolerance.
        self.ik_position_tolerance = 0.00005
        self.cartesian_step = 0.002
        self.cartesian_path_tolerance = 0.00025

        self.arm_indices = np.asarray(
            self.scene.getJointPositionIndices(self.joint_names),
            dtype=int,
        )
        self.viewer = None
        if show_viewer:
            self._create_viewer()

    def _create_viewer(self):
        import pinocchio as pin
        from pinocchio.visualize import ViserVisualizer

        self.viewer_model = pin.buildModelFromUrdf(str(self.urdf))
        for name in self.joint_names:
            index = self.viewer_model.joints[self.viewer_model.getJointId(name)].idx_q
            limits = self.scene.getJointInfo(name).limits
            self.viewer_model.lowerPositionLimit[index] = limits.min_position[0]
            self.viewer_model.upperPositionLimit[index] = limits.max_position[0]
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

    def preview_joint_offsets(self, q_measured, initial_offsets=None):
        """
        q_measured: full RoboPlan configuration, radians.
        Returns model-minus-servo offsets in degrees.
        Does not command the physical arm.
        """
        if self.viewer is None:
            raise RuntimeError("Initialize with show_viewer=True")

        offsets = {
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
        }
        if initial_offsets:
            offsets.update(initial_offsets)

        # Map the measured configuration into the viewer by joint name.
        baseline = pin.neutral(self.viewer_model)
        viewer_indices = {}

        for name in self.scene.getJointNames():
            source = self.scene.getJointPositionIndices([name])[0]
            joint_id = self.viewer_model.getJointId(name)
            destination = self.viewer_model.joints[joint_id].idx_q

            baseline[destination] = q_measured[source]
            viewer_indices[name] = destination

        sliders = {
            name: self.viewer.viewer.gui.add_slider(
                label=f"{name} offset (degrees)",
                min=-15.0,
                max=15.0,
                step=0.1,
                initial_value=float(offsets[name]),
            )
            for name in ("shoulder_lift", "elbow_flex")
        }

        def refresh(_=None):
            corrected = baseline.copy()

            for name, slider in sliders.items():
                corrected[viewer_indices[name]] += np.deg2rad(
                    slider.value
                )

            self.viewer.display(corrected)

        for slider in sliders.values():
            slider.on_update(refresh)

        self.viewer.displayVisuals(True)
        self.viewer.displayCollisions(False)
        refresh()

        try:
            input(
                "Adjust offsets in the browser to match the stationary arm.\n"
                "Press Enter here when finished.\n"
            )
            result = {
                name: float(slider.value)
                for name, slider in sliders.items()
            }
            print("Model-minus-servo offsets (degrees):", result)
            return result
        finally:
            for slider in sliders.values():
                slider.remove()

    def _sample_ik_seed(self, q_start, max_samples=100):
        """Sample arm joints within limits, preserving the fixed gripper."""
        for _ in range(max_samples):
            sampled = self.scene.randomPositions()
            q_seed = q_start.copy()
            q_seed[self.arm_indices] = sampled[self.arm_indices]
            if not self.scene.hasCollisions(q_seed):
                return q_seed
        return None

    def brush_tilt_deg(self, q):
        """Angle of the shaft toward the bristles from base-frame downward."""
        pose = self.scene.forwardKinematics(q, self.tip_frame, self.base_frame)
        axis = pose[:3, :3] @ self.brush_axis_local
        return float(np.rad2deg(np.arccos(np.clip(
            np.dot(axis, self.brush_direction), -1.0, 1.0,
        ))))

    def _validate_brush_tilt_path(self, configurations, require_upright=True):
        """Allow a tilted approach; once inside the 10-degree cone, stay inside."""
        previous = configurations[0]
        aligned = self.brush_tilt_deg(previous) <= self.brush_tilt_tolerance_deg
        for current in configurations[1:]:
            steps = max(1, int(np.ceil(np.max(np.abs(current - previous)) / 0.01)))
            for fraction in np.linspace(0.0, 1.0, steps + 1)[1:]:
                tilt = self.brush_tilt_deg(previous + fraction * (current - previous))
                if tilt <= self.brush_tilt_tolerance_deg:
                    aligned = True
                elif aligned:
                    raise RuntimeError(
                        f"Brush path leaves the {self.brush_tilt_tolerance_deg:g}° "
                        f"upright cone (tilt={tilt:.2f}°)"
                    )
            previous = current
        if require_upright and not aligned:
            raise RuntimeError("Brush path never reaches the upright tolerance")

    def _solve_xyz_attempt(self, xyz, seed, tasks, tilt_limit=None):
        """Run one numerical IK attempt; this does not move the robot."""
        tilt_limit = self.brush_tilt_tolerance_deg if tilt_limit is None else tilt_limit
        q = seed.copy()
        delta = np.zeros(self.ik.num_variables, dtype=np.float64)
        deadline = time.monotonic() + self.ik_attempt_timeout

        for _ in range(self.ik_max_iters):
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"IK attempt timed out (brush tilt={self.brush_tilt_deg(q):.2f}°, "
                    f"limit={tilt_limit:g}°)"
                )

            self.scene.setJointPositions(q)
            achieved = self.scene.forwardKinematics(
                q, self.tip_frame, self.base_frame
            )
            position_error = np.linalg.norm(achieved[:3, 3] - xyz)
            tilt = self.brush_tilt_deg(q)
            if (
                position_error <= self.ik_position_tolerance
                and tilt <= tilt_limit
            ):
                return q

            delta.fill(0.0)
            # Numerical iterates are not executed. Check collisions on the
            # candidate goal, its connection, and the final trajectory instead.
            self.ik.solveIk(
                self.scene, tasks, self.ik_limits, [],
                delta, 1e-6,
            )
            if not np.all(np.isfinite(delta)):
                raise RuntimeError("IK produced non-finite joint increments")

            largest_step = np.max(np.abs(delta))
            if largest_step > np.deg2rad(2):
                delta *= np.deg2rad(2) / largest_step

            full_delta = self.scene.toFullJointVelocities(self.group_name, delta)
            if not np.all(np.isfinite(full_delta)):
                raise RuntimeError("IK produced non-finite full-model increments")
            if np.max(np.abs(full_delta)) < 1e-10:
                raise RuntimeError("IK stalled before reaching the target")
            q = self.scene.integrate(q, full_delta)
            # The constrained solver can land just outside a bound. Correct
            # only its 1e-6 solver tolerance in IK iterates, never a
            # measured start or a materially out-of-range solution.
            for name, index in zip(self.joint_names, self.arm_indices):
                limits = self.scene.getJointInfo(name).limits
                clipped = np.clip(q[index], limits.min_position[0], limits.max_position[0])
                if abs(clipped - q[index]) <= 1e-6:
                    q[index] = clipped

        raise RuntimeError(
            f"IK did not converge: position error={position_error * 1000:.2f} mm, "
            f"brush tilt={tilt:.2f}° (limit={tilt_limit:g}°)"
        )

    def _validate_start_joint_limits(self, q_start, input_degrees=None):
        """Report the complete start state without changing any joint angles."""
        if not np.all(np.isfinite(q_start)):
            raise ValueError("IK start contains non-finite joint values")

        if self.scene.isValidConfiguration(q_start):
            return

        details = []
        for name in self.scene.getJointNames():
            index = self.scene.getJointPositionIndices([name])[0]
            angle = q_start[index]
            limits = self.scene.getJointInfo(name).limits
            lower, upper = limits.min_position[0], limits.max_position[0]
            line = f"  {name}: "
            if input_degrees is not None:
                line += (
                    f"input={float(input_degrees[name]):.9f} deg, "
                    f"offset={self.joint_offsets_deg.get(name, 0.0):+.9f} deg, "
                )
            line += (
                f"model={np.rad2deg(angle):.9f} deg, "
                f"allowed=[{np.rad2deg(lower):.9f}, "
                f"{np.rad2deg(upper):.9f}] deg"
            )
            if angle < lower:
                line += f" -- BELOW lower by {np.rad2deg(lower - angle):.9g} deg"
            elif angle > upper:
                line += f" -- ABOVE upper by {np.rad2deg(angle - upper):.9g} deg"
            details.append(line)

        raise ValueError("IK start violates joint limits:\n" + "\n".join(details))

    def _solve_xyz(self, xyz, seed, *, brush_direction=None, tilt_limit=None):
        """Find a goal with a clear connection from seed; full-model radians."""
        q_start = np.asarray(seed, dtype=float).copy()
        self._validate_start_joint_limits(q_start)

        if self.scene.hasCollisions(q_start):
            raise RuntimeError("Measured/current start configuration collides")

        self.scene.setJointPositions(q_start)

        target = self.scene.forwardKinematics(
            q_start, self.tip_frame, self.base_frame
        ).copy()
        target[:3, 3] = xyz

        shaft_target = target.copy()
        direction = self.brush_direction if brush_direction is None else brush_direction
        shaft_target[:3, 3] -= self.brush_axis_length * direction
        # Aim along the requested shaft direction; the ordinary goal is down.
        # Both are position tasks, so rotation about the shaft remains free.
        tasks = [FrameTask(
            self.ik, self.scene,
            CartesianConfiguration(self.base_frame, frame, pose),
            FrameTaskOptions(
                # Tip accuracy takes priority over aiming at exact zero tilt;
                # the separate acceptance/path checks still enforce the cone.
                position_cost=10.0 if frame == self.tip_frame else 1.0,
                orientation_cost=0.0, task_gain=0.5, lm_damping=0.001,
            ),
        ) for frame, pose in (
            (self.tip_frame, target), (self.shaft_frame, shaft_target),
        )]

        fixed_indices = np.setdiff1d(
            np.arange(q_start.size), self.arm_indices
        )
        last_failure = "No candidate found"
        try:
            for attempt in range(self.ik_max_attempts):
                q_seed = (
                    q_start.copy() if attempt == 0
                    else self._sample_ik_seed(q_start)
                )
                if q_seed is None:
                    last_failure = "Could not sample a collision-free seed"
                    print(f"IK attempt {attempt + 1}: {last_failure}", flush=True)
                    continue

                try:
                    options = {} if tilt_limit is None else {"tilt_limit": tilt_limit}
                    q_goal = self._solve_xyz_attempt(xyz, q_seed, tasks, **options)
                except RuntimeError as exc:
                    last_failure = f"IK failed: {exc}"
                    print(f"IK attempt {attempt + 1}: {last_failure}", flush=True)
                    continue

                if (
                    q_goal.shape != q_start.shape
                    or not np.all(np.isfinite(q_goal))
                    or not self.scene.isValidConfiguration(q_goal)
                ):
                    last_failure = "IK returned an invalid configuration"
                elif not np.allclose(
                    q_goal[fixed_indices], q_start[fixed_indices],
                    rtol=0.0, atol=1e-12,
                ):
                    raise RuntimeError("IK changed a joint outside the arm group")
                elif self.scene.hasCollisions(q_goal):
                    last_failure = "IK goal collides"
                elif hasCollisionsAlongPath(
                    self.scene, q_start, q_goal, max_step_size=0.01
                ):
                    # Always connect from the actual start, never the random seed.
                    last_failure = "Goal is clear, but the connecting path collides"
                else:
                    try:
                        options = {} if tilt_limit is None else {
                            "require_upright": tilt_limit <= self.brush_tilt_tolerance_deg,
                        }
                        self._validate_brush_tilt_path([q_start, q_goal], **options)
                    except RuntimeError as exc:
                        last_failure = str(exc)
                    else:
                        return q_goal.copy()

                print(f"IK attempt {attempt + 1}: {last_failure}", flush=True)

            raise RuntimeError(
                f"No collision-free connection to XYZ {xyz} after "
                f"{self.ik_max_attempts} attempts. Last failure: {last_failure}"
            )
        finally:
            self.scene.setJointPositions(q_start)

    def plan_xyz_path(self, positions, current_joints_deg, *, previous_command_deg=None):
        """
        positions: target XYZ points in robot base coordinates, meters.
        current_joints_deg: arm joint observations in degrees, before offsets;
                            gripper is already a MODEL angle in degrees.
        previous_command_deg: optional last completed arm command in servo
                              degrees, within 2 degrees of the measured pose.

        Returns a RoboPlan JointTrajectory. Does not move the arm.
        Follow the Cartesian polyline (0.25 mm model tolerance), stopping with
        zero velocity/acceleration at direction changes. Collinear points flow
        through without stops. Unreachable/unsafe straight paths are rejected.
        Goals must be within 10 degrees of bristles-down vertical. A tilted
        initial pose may approach the cone; after entering it, motion stays in it.
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
            offset = self.joint_offsets_deg.get(name, 0.0)
            q_full[indices[0]] = np.deg2rad(angle + offset)

        # Retain the original planner inputs in errors before updating the scene.
        self._validate_start_joint_limits(q_full, current_joints_deg)

        self.scene.setJointPositions(q_full)
        if self.scene.hasCollisions(q_full):
            raise RuntimeError("Measured/current start configuration collides")

        q = q_full.copy()
        if previous_command_deg is not None:
            # Preserve the last setpoint instead of stepping back to lagging
            # feedback between moves. Still validate the real measured start
            # and its connection to this setpoint in _validate_trajectory below.
            for name in self.joint_names:
                angle = float(previous_command_deg[name])
                if not np.isfinite(angle) or abs(angle - current_joints_deg[name]) > 2.0:
                    raise ValueError(f"Previous command for {name} differs from feedback by more than 2 degrees")
                index = self.scene.getJointPositionIndices([name])[0]
                q[index] = np.deg2rad(angle + self.joint_offsets_deg[name])
            self._validate_start_joint_limits(q)
            if hasCollisionsAlongPath(self.scene, q_full, q, max_step_size=0.01):
                raise RuntimeError("Planned trajectory intersects an obstacle")
        start_xyz = self.scene.forwardKinematics(q, self.tip_frame, self.base_frame)[:3, 3]
        corners = self._cartesian_corners(np.vstack((start_xyz, points)))
        if len(corners) == 1 and self.brush_tilt_deg(q) > self.brush_tilt_tolerance_deg:
            corners.append(corners[0].copy())  # Align in place if feasible.
        pieces = []
        try:
            for start_xyz, end_xyz in zip(corners, corners[1:]):
                piece = self._plan_cartesian_segment(start_xyz, end_xyz, q.copy())
                pieces.append(piece)
                q[self.arm_indices] = piece.positions[-1]
            if not pieces:
                return None
            trajectory = JointTrajectory()
            trajectory.joint_names = self.joint_names
            times, positions, velocities, accelerations = [], [], [], []
            elapsed = 0.0
            for piece in pieces:
                keep = slice(1, None) if times else slice(None)
                times.append((elapsed + np.asarray(piece.times))[keep])
                positions.append(np.asarray(piece.positions)[keep])
                velocities.append(np.asarray(piece.velocities)[keep])
                accelerations.append(np.asarray(piece.accelerations)[keep])
                elapsed += piece.times[-1]
            trajectory.times = np.concatenate(times)
            trajectory.positions = np.concatenate(positions)
            trajectory.velocities = np.concatenate(velocities)
            trajectory.accelerations = np.concatenate(accelerations)
            # Includes the measured pose -> previous command connection, and
            # enforces the tilt cone across the entire motion, not just each leg.
            self._validate_trajectory(trajectory, q_full)
            return trajectory
        finally:
            self.scene.setJointPositions(q_full)

    @staticmethod
    def _cartesian_corners(points):
        """Keep turns/reversals; merge only duplicate or same-direction points."""
        corners = []
        for point in points:
            point = np.asarray(point, dtype=float)
            if corners and np.linalg.norm(point - corners[-1]) < 1e-9:
                continue
            while len(corners) >= 2:
                before = corners[-1] - corners[-2]
                after = point - corners[-1]
                if np.linalg.norm(before / np.linalg.norm(before) - after / np.linalg.norm(after)) > 1e-7:
                    break
                corners.pop()
            corners.append(point.copy())
        return corners

    def _plan_cartesian_segment(self, start_xyz, end_xyz, q_start):
        """Quintic timing along one straight Cartesian leg, stopping at corners."""
        # Gradually align the shaft along the leg. Snapping the orientation to
        # vertical at its first 2 mm sample can curve even a straight tip path.
        pose = self.scene.forwardKinematics(q_start, self.tip_frame, self.base_frame)
        initial_axis = pose[:3, :3] @ self.brush_axis_local
        angle = np.arccos(np.clip(np.dot(initial_axis, self.brush_direction), -1.0, 1.0))
        axis = np.cross(initial_axis, self.brush_direction)
        if np.linalg.norm(axis) < 1e-10:
            axis = np.cross(initial_axis, np.eye(3)[np.argmin(abs(initial_axis))])
        axis /= np.linalg.norm(axis)
        count = max(1, int(np.ceil(np.linalg.norm(end_xyz - start_xyz) / self.cartesian_step)),
                    int(np.ceil(angle / np.deg2rad(2.0))))
        q = q_start.copy()
        waypoints = [q[self.arm_indices].copy()]
        for fraction in np.linspace(0.0, 1.0, count + 1)[1:]:
            xyz = start_xyz + fraction * (end_xyz - start_xyz)
            direction = Rotation.from_rotvec(axis * angle * fraction).apply(initial_axis)
            tilt_limit = (max(self.brush_tilt_tolerance_deg, np.rad2deg(angle))
                          if fraction < 1.0 else self.brush_tilt_tolerance_deg)
            q_next = self._solve_xyz(xyz, q, brush_direction=direction, tilt_limit=tilt_limit)
            if np.max(np.abs(q_next - q)) > 1e-8:
                waypoints.append(q_next[self.arm_indices].copy())
            q = q_next
        if len(waypoints) < 2:
            # A sub-tolerance move still has a well-defined stationary trajectory.
            waypoints.append(waypoints[0].copy())
        for stop_at_waypoints in (False, True):
            try:
                trajectory = self._quintic_trajectory(waypoints, stop_at_waypoints=stop_at_waypoints)
                self._validate_trajectory(trajectory, q_start)
                self._validate_cartesian_segment(trajectory, q_start, start_xyz, end_xyz)
                return trajectory
            except RuntimeError as exc:
                if stop_at_waypoints:
                    raise
                print(f"Cartesian segment cubic rejected: {exc}. Retrying with quintic waypoint stops.", flush=True)

    def _validate_cartesian_segment(self, trajectory, q_full, start, end):
        """Check the assigned finite line, including between command samples.

        A whole-polyline nearest-distance check can miss corner cutting or
        visiting the correct segments in the wrong order. Validate each leg
        separately, and reject backwards travel beyond the numerical tolerance.
        This is a model-space check, not a guarantee of physical tracking accuracy.
        """
        direction = end - start
        length = np.linalg.norm(direction)
        direction = direction / length if length > 0 else np.zeros(3)
        furthest = 0.0
        previous = np.asarray(trajectory.positions[0])
        for current in trajectory.positions:
            current = np.asarray(current)
            steps = max(1, int(np.ceil(np.max(np.abs(current - previous)) / 0.002)))
            for fraction in np.linspace(0.0, 1.0, steps + 1):
                q = q_full.copy()
                q[self.arm_indices] = previous + fraction * (current - previous)
                xyz = self.scene.forwardKinematics(q, self.tip_frame, self.base_frame)[:3, 3]
                progress = float(np.dot(xyz - start, direction))
                closest = start + np.clip(progress, 0.0, length) * direction
                error = np.linalg.norm(xyz - closest)
                if error > self.cartesian_path_tolerance or progress < furthest - self.cartesian_path_tolerance:
                    raise RuntimeError(
                        f"Planned trajectory leaves its Cartesian segment "
                        f"(line error={error * 1000:.3f} mm, "
                        f"tolerance={self.cartesian_path_tolerance * 1000:g} mm)"
                    )
                furthest = max(furthest, progress)
            previous = current
        if np.linalg.norm(xyz - end) > self.cartesian_path_tolerance:
            raise RuntimeError("Planned trajectory does not reach its Cartesian segment endpoint")

    def _quintic_trajectory(
        self, waypoints, stop_at_waypoints=False, dt=0.02,
        velocity_scale=0.15, acceleration_scale=0.225,
    ):
        """Time a smooth joint path with s(u)=10u³−15u⁴+6u⁵, u=t/T.

        Size T from continuous derivative bounds, not just sampled peaks.
        The fallback follows each checked straight segment, with zero velocity
        AND acceleration at its ends. The regular cubic path has no forced
        intermediate stops. No servo settings or hardware are involved.
        speed_scale compresses time: velocity scales by k, acceleration by k².
        The model's full velocity/acceleration limits remain upper bounds.
        """
        def clock(u):
            return (u**3 * (10 + u * (-15 + 6*u)),
                    30 * u**2 * (1-u)**2,
                    60 * u * (1-u) * (1-2*u))

        scales = np.array([dt, velocity_scale, acceleration_scale, self.speed_scale])
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("Quintic timing requires positive finite dt and motion limits")
        limits = [self.scene.getJointInfo(name).limits for name in self.joint_names]
        vmax = min(velocity_scale * self.speed_scale, 1.0) * np.array(
            [limit.max_velocity[0] for limit in limits]
        )
        amax = min(acceleration_scale * self.speed_scale**2, 1.0) * np.array(
            [limit.max_acceleration[0] for limit in limits]
        )
        if dt <= 0 or not np.isfinite(dt) or np.any(~np.isfinite(vmax)) or np.any(
            ~np.isfinite(amax)
        ) or np.any(vmax <= 0) or np.any(amax <= 0):
            raise ValueError("Quintic timing requires positive finite dt and motion limits")

        waypoints = np.asarray(waypoints, dtype=float)
        pieces = ([waypoints[i:i+2] for i in range(len(waypoints)-1)]
                  if stop_at_waypoints else [waypoints])
        times, positions, velocities, accelerations = [], [], [], []
        elapsed = 0.0
        for piece in pieces:
            # Chord length avoids assigning equal path length to a tiny IK
            # correction and a long stroke segment.
            knots = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(piece, axis=0), axis=1))]
            # A sub-tolerance Cartesian request can be a stationary hold.
            if knots[-1] == 0.0:
                knots = np.linspace(0.0, 1.0, len(piece))
            else:
                knots /= knots[-1]
            curve = CubicSpline(knots, piece, bc_type="natural")

            # Exact bounds on |dq/ds| and |d²q/ds²| within each cubic piece.
            h = np.diff(knots)[:, None]
            a, b, c = curve.c[:3]
            critical = np.clip(np.divide(-b, 3*a, out=np.zeros_like(a), where=a != 0), 0, h)
            first = np.maximum.reduce([
                abs(c), abs(3*a*h*h + 2*b*h + c),
                abs(3*a*critical**2 + 2*b*critical + c),
            ])
            second = np.maximum(abs(2*b), abs(6*a*h + 2*b))
            # Restrict the clock's derivative bounds to each piece's time
            # interval. Chain rule: q_ddot = q_ss*s_dot² + q_s*s_ddot.
            u_knots = np.array([brentq(lambda u: clock(u)[0] - s, 0, 1) for s in knots])
            lo, hi = u_knots[:-1], u_knots[1:]
            speed = clock(np.clip(0.5, lo, hi))[1][:, None]
            accel = np.maximum.reduce([
                abs(clock(lo)[2]), abs(clock(hi)[2]),
                abs(clock(np.clip((3-np.sqrt(3))/6, lo, hi))[2]),
                abs(clock(np.clip((3+np.sqrt(3))/6, lo, hi))[2]),
            ])[:, None]
            duration = max(
                float(np.max(first * speed / vmax)),
                float(np.sqrt(np.max((second * speed**2 + first * accel) / amax))),
            )
            # Whole command periods avoid a very short final send interval.
            steps = max(2, int(np.ceil(duration / dt)))
            duration = steps * dt
            t = np.linspace(0.0, duration, steps + 1)
            s, ds, dds = clock(t / duration)
            q = curve(s)
            v = curve(s, 1) * ds[:, None] / duration
            acc = (curve(s, 2) * ds[:, None]**2 + curve(s, 1) * dds[:, None]) / duration**2
            q[[0, -1]] = piece[[0, -1]]
            v[[0, -1]] = acc[[0, -1]] = 0.0
            keep = slice(1, None) if times else slice(None)
            times.append((elapsed + t)[keep])
            positions.append(q[keep])
            velocities.append(v[keep])
            accelerations.append(acc[keep])
            elapsed += duration

        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names
        trajectory.times = np.concatenate(times)
        trajectory.positions = np.concatenate(positions)
        trajectory.velocities = np.concatenate(velocities)
        trajectory.accelerations = np.concatenate(accelerations)
        return trajectory

    def _validate_trajectory(self, trajectory, q_full):
        """Apply the same limits, collision and tilt checks to either path."""
        if len(trajectory.times) == 0:
            raise RuntimeError("Joint trajectory parameterization failed")

        # Validate the returned motion before allowing execution.
        output_indices = self.scene.getJointPositionIndices(
            trajectory.joint_names
        )
        previous = q_full.copy()
        configurations = [previous]
        for q in trajectory.positions:
            current = q_full.copy()
            current[output_indices] = q
            if not self.scene.isValidConfiguration(current):
                raise RuntimeError("Planned trajectory violates calibrated joint limits")
            if hasCollisionsAlongPath(
                self.scene, previous, current, max_step_size=0.01
            ):
                raise RuntimeError("Planned trajectory intersects an obstacle")
            previous = current
            configurations.append(current)

        self._validate_brush_tilt_path(configurations)
