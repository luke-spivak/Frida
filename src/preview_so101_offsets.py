#!/usr/bin/env python3
"""Inspect a stationary SO-101 in Viser using read-only motor snapshots.

Run from src: python preview_so101_offsets.py
No SO101Follower/Painter connection, torque changes, calibration writes, or
position commands. Each snapshot opens the bus, reads, and closes it again.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import webbrowser

import numpy as np
import pinocchio as pin
from pinocchio.visualize import ViserVisualizer
import viser

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


ARM_JOINTS = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll",
)
ALL_JOINTS = (*ARM_JOINTS, "gripper")
DEFAULT_OFFSETS = dict.fromkeys(ARM_JOINTS, 0.0)
DEFAULT_OFFSETS["elbow_flex"] = -4.6  # Initial guess before a saved calibration exists.
DEFAULT_OFFSETS_PATH = Path(__file__).resolve().with_name("so101_offsets.json")
DEFAULT_URDF = Path(__file__).resolve().parent / "SO101/so101_brush_tip_nominal.urdf"
DEFAULT_CALIBRATION = (
    Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/my_arm.json"
)


class ReadOnlyFeetechBus(FeetechMotorsBus):
    """Block the low-level register-write paths, including torque/calibration."""

    def _write(self, *args, **kwargs):
        raise RuntimeError("Motor register writes are disabled in this read-only tool")

    def _sync_write(self, *args, **kwargs):
        raise RuntimeError("Motor register writes are disabled in this read-only tool")


def validate_observation(observation):
    result = {f"{name}.pos": float(observation[f"{name}.pos"]) for name in ALL_JOINTS}
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("Joint readings must be finite")
    if not 0.0 <= result["gripper.pos"] <= 100.0:
        raise ValueError("gripper.pos must be a LeRobot opening in [0, 100]")
    return result


def read_motor_snapshot(port, calibration_path):
    saved = json.loads(Path(calibration_path).expanduser().read_text())
    calibration = {name: MotorCalibration(**saved[name]) for name in ALL_JOINTS}
    motors = {
        name: Motor(
            calibration[name].id, "sts3215",
            MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.DEGREES,
        )
        for name in ALL_JOINTS
    }
    bus = ReadOnlyFeetechBus(port, motors, calibration=calibration)
    try:
        # The bus handshake only pings/reads. SO101Follower.connect() additionally
        # configures motors and toggles torque, so it must not be used here.
        bus.connect()
        if not bus.is_calibrated:
            raise RuntimeError(
                "Saved calibration does not match motor registers. "
                "Select the matching --calibration file; no registers were changed."
            )
        positions = bus.sync_read("Present_Position", normalize=True, num_retry=2)
        return validate_observation({f"{name}.pos": value for name, value in positions.items()})
    finally:
        if bus.is_connected:
            # The default disconnect disables torque; explicitly preserve it.
            bus.disconnect(disable_torque=False)


def load_observation(path):
    saved = json.loads(Path(path).expanduser().read_text())
    return validate_observation(saved.get("observation", saved))


def joint_index(model, name):
    if not model.existJointName(name):
        raise ValueError(f"URDF has no joint named {name}")
    joint = model.joints[model.getJointId(name)]
    if joint.nq != 1:
        raise ValueError(f"Expected a bounded revolute joint: {name}")
    return joint.idx_q


def model_configuration(model, observation, offsets):
    """Convert by name without clamping: out-of-limit poses must remain visible."""
    q = pin.neutral(model)
    for name in ARM_JOINTS:
        q[joint_index(model, name)] = np.deg2rad(observation[f"{name}.pos"] + offsets[name])
    index = joint_index(model, "gripper")
    lower, upper = model.lowerPositionLimit[index], model.upperPositionLimit[index]
    # Same nominal opening-to-model mapping as robot.py, not a jaw calibration.
    q[index] = lower + observation["gripper.pos"] / 100.0 * (upper - lower)
    return q


def pose_table(model, observation, offsets, q):
    rows = [
        "| Joint | Read (deg) | Offset (deg) | Model (deg) | URDF range (deg) |",
        "|---|---:|---:|---:|---|",
    ]
    outside = []
    for name in ARM_JOINTS:
        index = joint_index(model, name)
        angle = float(np.rad2deg(q[index]))
        lower, upper = np.rad2deg([model.lowerPositionLimit[index], model.upperPositionLimit[index]])
        rows.append(
            f"| {name} | {observation[f'{name}.pos']:.2f} | {offsets[name]:+.2f} "
            f"| {angle:.2f} | {lower:.2f} to {upper:.2f} |"
        )
        if q[index] < model.lowerPositionLimit[index] or q[index] > model.upperPositionLimit[index]:
            outside.append(name)
    if outside:
        rows.append("\n**Outside URDF limits:** " + ", ".join(outside) + ". Displayed without clamping.")
    else:
        rows.append("\nAll arm angles are within URDF limits. This is not a collision check.")
    gripper_deg = np.rad2deg(q[joint_index(model, "gripper")])
    rows.append(
        f"\nGripper opening: {observation['gripper.pos']:.3f}/100; "
        f"nominal model angle: {gripper_deg:.3f}°. Gripper has no offset slider."
    )
    return "\n".join(rows)


class OffsetViewer:
    def __init__(self, args, read_snapshot, initial_offsets):
        self.args = args
        self.read_snapshot = read_snapshot
        self.lock = threading.RLock()
        self.initial_offsets = initial_offsets.copy()
        # Finish all serial access before starting the viewer.
        self.observation = read_snapshot()
        self.captured_at = datetime.now(timezone.utc).isoformat()
        self.model = pin.buildModelFromUrdf(str(args.urdf))
        package_dirs = [str(args.urdf.parent)]
        collision = pin.buildGeomFromUrdf(
            self.model, str(args.urdf), pin.GeometryType.COLLISION, package_dirs=package_dirs,
        )
        visual = pin.buildGeomFromUrdf(
            self.model, str(args.urdf), pin.GeometryType.VISUAL, package_dirs=package_dirs,
        )
        # Validate the URDF joint mapping before opening a server.
        model_configuration(self.model, self.observation, initial_offsets)
        self.server = viser.ViserServer(host="127.0.0.1", port=args.viewer_port)
        self.visualizer = ViserVisualizer(self.model, collision, visual)
        # Pass a server explicitly: this Pinocchio version passes the obsolete
        # server_port keyword when creating one, otherwise ignoring our port.
        self.visualizer.initViewer(viewer=self.server, open=False, loadModel=True)
        self.server.scene.set_up_direction("+z")

        @self.server.on_client_connect
        def set_camera(client):
            client.camera.position = (0.55, 0.55, 0.4)
            client.camera.look_at = (0.0, 0.05, 0.13)
            client.camera.up_direction = (0.0, 0.0, 1.0)

        self.server.gui.add_markdown(
            "## SO-101 zero offsets\n"
            "**Read-only snapshot. Sliders move the model only.**\n\n"
            "Model angle = observed arm angle + offset. Match the link poses, "
            "then verify the same offsets against other measured poses. "
            "Changing a slider does not change the robot or its torque state."
        )
        self.status = self.server.gui.add_markdown("")
        self.sliders = {
            name: self.server.gui.add_slider(
                f"{name} offset (deg)", min=-args.offset_range, max=args.offset_range,
                step=0.1, initial_value=initial_offsets[name],
            )
            for name in ARM_JOINTS
        }
        self.show_collisions = self.server.gui.add_checkbox("Show collision meshes", False)
        refresh = self.server.gui.add_button("Refresh measured pose")
        reset = self.server.gui.add_button("Reset offsets to initial values")
        save = self.server.gui.add_button("Save snapshot and offsets")
        self.table = self.server.gui.add_markdown("")
        self.export = self.server.gui.add_markdown("")

        for slider in self.sliders.values():
            slider.on_update(self.render)
        self.show_collisions.on_update(self.render)
        refresh.on_click(self.refresh)
        reset.on_click(self.reset)
        save.on_click(self.save)
        self.status.content = f"Snapshot captured: {self.captured_at}"
        self.render()

    def offsets(self):
        return {name: float(slider.value) for name, slider in self.sliders.items()}

    def render(self, _event=None):
        with self.lock:
            offsets = self.offsets()
            q = model_configuration(self.model, self.observation, offsets)
            self.visualizer.displayVisuals(not self.show_collisions.value)
            self.visualizer.displayCollisions(self.show_collisions.value)
            self.visualizer.display(q)
            self.table.content = pose_table(self.model, self.observation, offsets, q)
            self.export.content = "Planner offsets (not applied):\n```json\n" + json.dumps(offsets, indent=2) + "\n```"

    def refresh(self, _event=None):
        with self.lock:
            try:
                observation = self.read_snapshot()
            except Exception as error:
                self.status.content = f"**Read failed; retaining previous snapshot:** {error}"
                return
            self.observation = observation
            self.captured_at = datetime.now(timezone.utc).isoformat()
            self.status.content = f"Snapshot captured: {self.captured_at}"
            self.render()

    def reset(self, _event=None):
        with self.lock:
            for name, slider in self.sliders.items():
                slider.value = self.initial_offsets[name]
            self.render()

    def save(self, _event=None):
        with self.lock:
            record = {
                "joint_offsets_deg": self.offsets(),
                "observation": self.observation,
                "observation_units": "arm: degrees; gripper: normalized 0-100 opening",
                "captured_at": self.captured_at,
                "urdf": str(self.args.urdf),
                "calibration": str(self.args.calibration),
                "source": str(self.args.observation) if self.args.observation else self.args.port,
            }
            try:
                self.args.output.write_text(json.dumps(record, indent=2) + "\n")
            except OSError as error:
                self.status.content = f"**Could not save:** {error}"
                return
            self.status.content = f"Saved to `{self.args.output}`. Planner settings are unchanged."
            print(f"Saved {self.args.output}\n{json.dumps(record, indent=2)}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/tty.usbmodem5B8E1155331")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--viewer-port", type=int, default=8081)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--observation", type=Path, help="Offline mode: read a saved observation JSON; no serial access")
    parser.add_argument(
        "--initial-offsets", type=Path,
        default=DEFAULT_OFFSETS_PATH if DEFAULT_OFFSETS_PATH.exists() else None,
        help="Load saved offsets (defaults to src/so101_offsets.json when present)",
    )
    parser.add_argument("--offset", action="append", default=[], metavar="JOINT=DEGREES")
    parser.add_argument("--offset-range", type=float, default=30.0, help="Slider range is +/- this many degrees")
    parser.add_argument("--output", type=Path, default=DEFAULT_OFFSETS_PATH, help="JSON destination for the Save button (default: src/so101_offsets.json)")
    args = parser.parse_args(argv)
    for name in ("calibration", "urdf", "observation", "initial_offsets", "output"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    offsets = DEFAULT_OFFSETS.copy()
    if args.initial_offsets:
        saved = json.loads(args.initial_offsets.read_text())
        offsets.update(saved.get("joint_offsets_deg", saved))
    for item in args.offset:
        try:
            name, value = item.split("=", 1)
            offsets[name] = float(value)
        except ValueError:
            parser.error("Offsets must have the form --offset shoulder_lift=3.6")
    if set(offsets) != set(ARM_JOINTS):
        parser.error("Offsets may only name the five arm joints")
    if not np.isfinite(args.offset_range) or args.offset_range <= 0:
        parser.error("--offset-range must be finite and positive")
    offsets = {name: float(value) for name, value in offsets.items()}
    if not all(np.isfinite(value) and abs(value) <= args.offset_range for value in offsets.values()):
        parser.error("Initial offsets must be finite and within --offset-range")
    if args.output in {args.calibration, args.urdf, args.observation}:
        parser.error("--output must not overwrite the calibration, URDF, or observation input")
    return args, offsets


def main(argv=None):
    args, offsets = parse_args(argv)
    if args.observation:
        reader = lambda: load_observation(args.observation)
        print(f"Offline observation: {args.observation}", flush=True)
    else:
        reader = lambda: read_motor_snapshot(args.port, args.calibration)
        print("Reading motor positions only; torque and calibration will not be changed.", flush=True)
    viewer = OffsetViewer(args, reader, offsets)
    url = f"http://{viewer.server.get_host()}:{viewer.server.get_port()}"
    print(f"Offset viewer: {url}\nUse Save to export. Ctrl-C to exit.", flush=True)
    try:
        if not args.no_browser:
            webbrowser.open(url)
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        print("Final offsets (not applied):\n" + json.dumps(viewer.offsets(), indent=2), flush=True)
        viewer.server.stop()


if __name__ == "__main__":
    main()
