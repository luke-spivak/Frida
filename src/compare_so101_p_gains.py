#!/usr/bin/env python3
"""Compare P=32, 40, 48, 56 on each of the five SO101 arm joints, one at a time.

Run from src: python compare_so101_p_gains.py --execute
Without --execute this only prints settings, without loading hardware drivers.
Start in clear air: direct servo commands have no IK/collision checking. Other
joints (including the gripper) hold their measured pose. Only P is tuned: I, D,
dead zones, and startup force are read and verified, never written. Each joint's
original P is restored before testing the next joint. Cleanup restores original
P and torque/lock states, including on Ctrl-C. Initially unpowered joints relax
again on exit. Do not run another bus client. Each test requests a tiny step
from the measured joint angle and holds that target fixed for 2.5 seconds,
then returns toward the original center. Use --joints to select fewer joints.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time


ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
P_GAINS = (32, 40, 48, 56)
STARTUP_REGISTER = "Minimum_Startup_Force"
ADJACENT_REGISTER = "Register_25"
UNCHANGED_REGISTERS = ("I_Coefficient", "D_Coefficient", "CW_Dead_Zone",
                       "CCW_Dead_Zone", STARTUP_REGISTER, ADJACENT_REGISTER)
JOINTS = (*ARM_JOINTS, "gripper")
ROOT = Path(__file__).resolve().parent.parent
CALIBRATION = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/my_arm.json"
REGISTERS = (
    "Firmware_Major_Version", "Firmware_Minor_Version", "Angular_Resolution",
    "P_Coefficient", "I_Coefficient", "D_Coefficient", "CW_Dead_Zone",
    "CCW_Dead_Zone", STARTUP_REGISTER, ADJACENT_REGISTER,
    "Acceleration", "Goal_Velocity", "Torque_Limit",
    "Operating_Mode", "Torque_Enable", "Lock", "Goal_Position", "Present_Position",
)
FEEDBACK = {
    "position": "Present_Position", "drive_output": "Present_Load",
    "current": "Present_Current", "voltage": "Present_Voltage",
}
FIELDS = (
    "command", "joint", "block", "startup_force", "cw_dead_zone", "ccw_dead_zone",
    "p_gain", "i_gain", "d_gain", "repeat", "phase",
    "requested_step_ticks", "initial_error_ticks", "target_offset_ticks",
    "previous_goal_raw", "command_step_ticks",
    "baseline_raw", "baseline_error_ticks", "baseline_read_end_s",
    "target_raw", "target_deg", "actual_raw", "actual_deg", "error_ticks",
    "error_deg", "displacement_ticks", "displacement_deg",
    "drive_output_raw", "drive_output_pct", "current_raw", "current_ma",
    "voltage_raw", "voltage_v", "command_start_s", "command_end_s",
    "read_start_s", "read_end_s",
    "elapsed_s", "read_duration_ms", "sample_interval_ms",
) + tuple(f"{name}_read_{edge}_s" for name in FEEDBACK for edge in ("start", "end"))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="connect, sweep P only, and move each selected joint")
    parser.add_argument("--port", default="/dev/tty.usbmodem5B8E1155331")
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument("--joints", nargs="+", choices=ARM_JOINTS, default=list(ARM_JOINTS),
                        help="arm joints to test in sequence; defaults to all five (never the gripper)")
    parser.add_argument("--step-ticks", type=int, nargs="+", default=[1, 2, 4],
                        help="step sizes, each tested in both directions (1-8 encoder ticks)")
    parser.add_argument("--repeats", type=int, default=3, help="repetitions of each size/direction per P gain per joint")
    parser.add_argument("--hold-s", type=float, default=2.5, help="fixed-target hold time (2-3 seconds)")
    parser.add_argument("--return-s", type=float, default=2.5, help="hold time at the original center between trials")
    parser.add_argument("--sample-hz", type=float, default=50.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/so101_p_gains")
    args = parser.parse_args(argv)
    if len(set(args.joints)) != len(args.joints):
        parser.error("joints must not contain duplicates")
    if any(not 1 <= step <= 8 for step in args.step_ticks) or args.repeats < 1:
        parser.error("step-ticks must be in [1, 8]; repeats must be positive")
    if len(set(args.step_ticks)) != len(args.step_ticks):
        parser.error("step-ticks must not contain duplicates")
    if not math.isfinite(args.hold_s) or not 2 <= args.hold_s <= 3:
        parser.error("hold-s must be in [2, 3] seconds")
    if not math.isfinite(args.return_s) or args.return_s < 0.5:
        parser.error("return-s must be at least 0.5 seconds")
    if not math.isfinite(args.sample_hz) or not 5 <= args.sample_hz <= 100:
        parser.error("sample-hz must be in [5, 100]")
    return args


def trial_steps(sizes):
    """Each size in both directions; a separate center return follows every step."""
    return [direction * size for size in sizes for direction in (1, -1)]


def summarize(rows):
    # Use the actual starting error, not the previous goal (which may be unreached).
    error = rows[0]["initial_error_ticks"]
    direction = (error > 0) - (error < 0)
    delays = {}
    for threshold in (1, 2):
        delays[f"movement_delay_{threshold}_ticks_s"] = next((
            a["elapsed_s"] for a, b in zip(rows, rows[1:])
            if direction and all(direction * row["displacement_ticks"] >= threshold for row in (a, b))
        ), None)
    window = min(0.25, rows[-1]["elapsed_s"] / 2)
    tail = [row for row in rows if row["elapsed_s"] >= rows[-1]["elapsed_s"] - window]
    return {
        **delays,
        "observation_duration_s": rows[-1]["elapsed_s"],
        "final_error_deg": rows[-1]["error_deg"],
        "tail_window_s": window,
        "tail_mean_error_deg": statistics.mean(row["error_deg"] for row in tail),
        "tail_mean_abs_error_deg": statistics.mean(abs(row["error_deg"]) for row in tail),
        "tail_motion_range_ticks": max(r["actual_raw"] for r in tail) - min(r["actual_raw"] for r in tail),
        "tail_mean_current_ma": statistics.mean(row["current_ma"] for row in tail),
        "tail_mean_drive_output_pct": statistics.mean(row["drive_output_pct"] for row in tail),
        "rms_error_deg": math.sqrt(statistics.mean(row["error_deg"] ** 2 for row in rows)),
        "peak_overshoot_deg": max(0.0, max(-direction * row["error_deg"] for row in rows)),
        "achieved_displacement_deg": rows[-1]["displacement_deg"],
        "write_duration_ms": 1000 * (rows[0]["command_end_s"] - rows[0]["command_start_s"]),
        "max_sample_interval_ms": max(row["sample_interval_ms"] for row in rows),
    }


class StepRecorder:
    def __init__(self, bus, joint, calibration, args, writer, center, deg_per_tick, epoch):
        self.bus, self.args, self.writer = bus, args, writer
        self.joint = joint
        self.calibration, self.center = calibration, center
        self.deg_per_tick = deg_per_tick
        self.mid = (calibration.range_min + calibration.range_max) / 2
        self.epoch = epoch
        # Include a little overshoot, but stop trials if the arm leaves this window.
        self.envelope_ticks = 2 * max(args.step_ticks) + 3

    def read(self, register, joint=None):
        return int(self.bus.read(register, joint or self.joint, normalize=False, num_retry=1))

    def write(self, register, value, joint=None):
        self.bus.write(register, joint or self.joint, int(value), normalize=False)

    def angle(self, raw):
        return (raw - self.mid) * self.deg_per_tick

    def check(self, raw):
        if not self.calibration.range_min <= raw <= self.calibration.range_max:
            raise RuntimeError(f"{self.joint} position {raw} is outside calibrated limits")
        if abs(raw - self.center) > self.envelope_ticks:
            raise RuntimeError(f"{self.joint} position {raw} is outside the small test window")

    def sample_step(self, offset, duration, command, block, settings, repeat, phase):
        previous_goal = self.read("Goal_Position")
        baseline = self.read("Present_Position")
        baseline_time = time.perf_counter() - self.epoch
        self.check(baseline)
        # A return can stop short. Anchor each isolated test to the
        # measured angle so its initial error really is +/-1, 2, or 4 ticks.
        # Return/precondition goals stay relative to the original center, so
        # small residual errors cannot accumulate into unbounded travel.
        target = (baseline if phase == "step" else self.center) + offset
        self.check(target)
        metadata = {
            "command": command, "joint": self.joint, "block": block,
            "p_gain": settings["P_Coefficient"], "i_gain": settings["I_Coefficient"],
            "d_gain": settings["D_Coefficient"], "startup_force": settings[STARTUP_REGISTER],
            "cw_dead_zone": settings["CW_Dead_Zone"], "ccw_dead_zone": settings["CCW_Dead_Zone"],
            "repeat": repeat, "phase": phase,
            "requested_step_ticks": offset if phase == "step" else None,
            "initial_error_ticks": target - baseline, "target_offset_ticks": target - self.center,
            "previous_goal_raw": previous_goal, "command_step_ticks": target - previous_goal,
            "baseline_raw": baseline, "baseline_error_ticks": previous_goal - baseline,
            "baseline_read_end_s": baseline_time, "target_raw": target, "target_deg": self.angle(target),
        }
        start = time.perf_counter()
        self.write("Goal_Position", target)
        end = time.perf_counter()
        if self.read("Goal_Position") != target:
            raise RuntimeError("Goal_Position readback differs from requested raw target")
        rows, previous = [], None
        period = 1 / self.args.sample_hz
        next_read = end
        while True:
            time.sleep(max(0, next_read - time.perf_counter()))
            read_start = time.perf_counter()
            row, values = dict(metadata), {}
            for name, register in FEEDBACK.items():
                row[f"{name}_read_start_s"] = time.perf_counter() - self.epoch
                values[name] = self.read(register)
                row[f"{name}_read_end_s"] = time.perf_counter() - self.epoch
            read_end = time.perf_counter()
            actual = values["position"]
            position_time = self.epoch + row["position_read_end_s"]
            row.update(
                actual_raw=actual, actual_deg=self.angle(actual), error_ticks=target - actual,
                error_deg=(target - actual) * self.deg_per_tick, displacement_ticks=actual - baseline,
                displacement_deg=(actual - baseline) * self.deg_per_tick,
                drive_output_raw=values["drive_output"], drive_output_pct=values["drive_output"] * 0.1,
                current_raw=values["current"], current_ma=values["current"] * 6.5,
                voltage_raw=values["voltage"], voltage_v=values["voltage"] * 0.1,
                command_start_s=start - self.epoch, command_end_s=end - self.epoch,
                read_start_s=read_start - self.epoch, read_end_s=read_end - self.epoch,
                elapsed_s=position_time - end, read_duration_ms=(read_end - read_start) * 1000,
                sample_interval_ms=0.0 if previous is None else (position_time - previous) * 1000,
            )
            self.writer.writerow(row)
            rows.append(row)
            self.check(actual)
            previous = position_time
            if position_time - end >= duration:
                return rows
            next_read += period
            if next_read < read_end:
                next_read = read_end + period  # No catch-up bursts after a slow read.


def execute(args):
    # Lazy imports: the default preview does not even load a hardware driver.
    # LeRobot imports torch indirectly. Load NumPy first so this macOS conda
    # environment shares one libomp instead of aborting during torch's import.
    import numpy  # noqa: F401 -- intentional import order

    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    saved = json.loads(args.calibration.expanduser().read_text())
    calibration = {name: MotorCalibration(**saved[name]) for name in JOINTS}
    motors = {name: Motor(calibration[name].id, "sts3215",
                          MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.DEGREES)
              for name in JOINTS}
    bus = FeetechMotorsBus(args.port, motors, calibration=calibration)
    run_comparison(args, bus, calibration)


def run_comparison(args, bus, calibration):
    """Run the sweep on a supplied bus (also supports fully fake offline tests)."""
    # The installed LeRobot table combines addresses 24-25. Override only this
    # bus instance to log both bytes separately. Neither is written by this test.
    bus.model_ctrl_table = {model: dict(table) for model, table in bus.model_ctrl_table.items()}
    bus.model_ctrl_table["sts3215"].update({
        STARTUP_REGISTER: (24, 1), ADJACENT_REGISTER: (25, 1),
    })
    output = args.output_dir.expanduser() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "experiment": "arm_joint_p_sweep", "schema_version": 5,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "joints": args.joints, "p_gain_sequence": P_GAINS,
        "unchanged_registers": UNCHANGED_REGISTERS,
        "blocks": [], "commands": [], "status": "starting", "restoration_errors": [],
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    original, recorder, changed, touched = {}, None, False, set()

    def verify_unchanged(name):
        values = {register: recorder.read(register, name) for register in UNCHANGED_REGISTERS}
        for register, value in values.items():
            if value != original[name][register]:
                raise RuntimeError(f"{name} {register} changed unexpectedly: {original[name][register]} -> {value}")
        return values

    def hold_joint(name):
        # Hold the measured position, never replay a stale goal during cleanup.
        position = recorder.read("Present_Position", name)
        if not calibration[name].range_min <= position <= calibration[name].range_max:
            raise RuntimeError(f"Cannot hold {name}: measured position is outside calibrated limits")
        recorder.write("Goal_Position", position, name)
        if recorder.read("Goal_Position", name) != position:
            raise RuntimeError(f"{name} hold goal readback failed")

    def set_p_gain(name, value):
        touched.add(name)  # Even a failed write may have reached the servo.
        # Keep torque on while changing P so a loaded shoulder/elbow cannot drop.
        # Unlock the parameter write, verify it, then relock. No other gain writes.
        recorder.write("Lock", 0, name)
        recorder.write("P_Coefficient", value, name)
        if recorder.read("P_Coefficient", name) != value:
            raise RuntimeError(f"{name} P_Coefficient readback failed")
        recorder.write("Lock", 1, name)

    try:
        bus.connect()  # Handshake only; no SO101Follower.configure() or calibration writes.
        if not bus.is_calibrated:
            raise RuntimeError("Saved calibration does not match motor registers; no settings changed")
        original = {name: {register: int(bus.read(register, name, normalize=False))
                           for register in REGISTERS} for name in JOINTS}
        report["original_registers"] = original
        if any(state["Operating_Mode"] != 0 for state in original.values()):
            raise RuntimeError("All joints must already be in position mode; no settings changed")
        for name in JOINTS:
            if not calibration[name].range_min <= original[name]["Present_Position"] <= calibration[name].range_max:
                raise RuntimeError(f"{name} is outside its calibrated range; no settings changed")
        if any(original[name]["Angular_Resolution"] != 1 for name in args.joints):
            raise RuntimeError("Expected angular resolution 1 on each tested joint; no settings changed")
        deg_per_tick = 360 / (bus.model_resolution_table["sts3215"] - 1)
        report.update(
            deg_per_tick=deg_per_tick,
            center_raw={name: original[name]["Present_Position"] for name in args.joints},
            trial_steps=trial_steps(args.step_ticks),
            target_definition="Step targets are measured baseline + requested signed ticks, held fixed; returns/preconditions use the original center. No assumption that a center return reaches zero error or resets integral state.",
            delay_definition="Directed displacement from the pre-command reading sustained for 2 samples; separate 1/2 tick thresholds; null if undetected within that command",
            timing_uncertainty="Write duration plus feedback interval; each serial read has its own timestamps; readings are not simultaneous",
            feedback_units={"drive_output_raw": "LeRobot sign-decoded Present_Load; 0.1% duty cycle/count, not joint torque",
                            "current_raw": "Present_Current; nominal 6.5 mA/count",
                            "voltage_raw": "Present_Voltage; 0.1 V/count"},
            register_reference="https://ftservo.github.io/ftservo-wiki/reference/parameter/memory-sts/",
        )
        save()  # Persist original settings before any write.
        with (output / "samples.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            epoch = time.perf_counter()
            recorders = {name: StepRecorder(bus, name, calibration[name], args, writer,
                                            original[name]["Present_Position"], deg_per_tick, epoch)
                         for name in args.joints}
            # Check every selected joint before any writes or enabling torque.
            for recorder in recorders.values():
                recorder.check(recorder.center - recorder.envelope_ticks)
                recorder.check(recorder.center + recorder.envelope_ticks)
            changed = True
            # Seed every hold goal before enabling torque, avoiding stale targets.
            for name in JOINTS:
                hold_joint(name)
            for name in JOINTS:
                if original[name]["Torque_Enable"] != 1:
                    bus.enable_torque(name)
            command = 0

            def record(offset, duration, repeat, phase):
                nonlocal command
                command += 1
                report["active_command"] = dict(command=command, joint=joint, block=block, p_gain=p_gain,
                                                repeat=repeat, phase=phase, requested_offset_ticks=offset)
                rows = recorder.sample_step(offset, duration, command, block, settings, repeat, phase)
                result = {key: rows[0][key] for key in (
                    "command", "joint", "block", "startup_force", "p_gain", "i_gain", "d_gain",
                    "cw_dead_zone", "ccw_dead_zone", "repeat", "phase",
                    "requested_step_ticks", "initial_error_ticks", "target_offset_ticks", "target_raw",
                    "previous_goal_raw", "command_step_ticks", "baseline_raw", "baseline_error_ticks",
                )}
                result.update(summarize(rows))
                report.pop("active_command")
                report["commands"].append(result)
                stream.flush()
                return result

            block = 0
            for joint in args.joints:
                recorder = recorders[joint]
                for p_gain in P_GAINS:
                    block += 1
                    recorder.check(recorder.read("Present_Position"))
                    hold_joint(joint)
                    set_p_gain(joint, p_gain)
                    settings = {"P_Coefficient": recorder.read("P_Coefficient"), **verify_unchanged(joint)}
                    report["blocks"].append({"block": block, "joint": joint, "p_gain": p_gain,
                                             "verified_settings": settings})
                    save()
                    # Same approach direction in every block; don't assume zero residual error.
                    record(max(args.step_ticks), args.return_s, 0, "precondition")
                    record(0, args.return_s, 0, "precondition")
                    for repeat in range(1, args.repeats + 1):
                        results = []
                        for step in trial_steps(args.step_ticks):
                            results.append(record(step, args.hold_s, repeat, "step"))
                            save()
                            record(0, args.return_s, repeat, "return")
                            save()
                        error = statistics.mean(result["tail_mean_abs_error_deg"] for result in results)
                        print(f"{joint}, P={p_gain}, repeat {repeat}/{args.repeats}: "
                              f"mean absolute tail error={error:.3f} deg")
                hold_joint(joint)
                set_p_gain(joint, original[joint]["P_Coefficient"])
                verify_unchanged(joint)
            report["status"] = "complete"
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if changed and not bus.is_connected:
            report["restoration_errors"].append("Bus disconnected before settings could be restored")
        if changed and bus.is_connected:
            def restore(label, action):
                try:
                    action()
                    return True
                except Exception as exc:
                    report["restoration_errors"].append(f"{label}: {exc}")
                    return False

            ready = dict.fromkeys(JOINTS, True)
            for name in JOINTS:
                if name in touched:
                    ready[name] = restore(f"hold current {name}", lambda name=name: hold_joint(name))
                    ready[name] = restore(f"restore {name} P", lambda name=name:
                                          set_p_gain(name, original[name]["P_Coefficient"])) and ready[name]
                ready[name] = restore(f"verify unchanged {name} settings", lambda name=name:
                                      verify_unchanged(name)) and ready[name]
            for name in JOINTS:
                # Do not re-enable a joint with an unverified controller or stale target.
                torque = original[name]["Torque_Enable"] if ready[name] else 0
                restore(f"restore {name} torque", lambda name=name, torque=torque: recorder.write("Torque_Enable", torque, name))
                restore(f"restore {name} lock", lambda name=name: recorder.write("Lock", original[name]["Lock"], name))
            report["restored_registers"] = {}
            for name in JOINTS:
                report["restored_registers"][name] = {}
                for register in ("Torque_Enable", "Lock", "P_Coefficient", *UNCHANGED_REGISTERS):
                    def verify(name=name, register=register):
                        actual = recorder.read(register, name)
                        report["restored_registers"][name][register] = actual
                        if actual != original[name][register]:
                            raise RuntimeError(f"Expected {original[name][register]}, read {actual}")
                    restore(f"verify {name} {register}", verify)
        try:
            if bus.is_connected:
                bus.disconnect(disable_torque=False)
        finally:
            if report["restoration_errors"]:
                report["status"] = "restoration_failed"
                print("RESTORATION FAILED: " + "; ".join(report["restoration_errors"]))
            save()
            print(f"Diagnostic files: {output}")
        if report["restoration_errors"]:
            raise RuntimeError("Could not restore every motor setting; see report.json")


def main():
    args = arguments()
    blocks = len(args.joints) * len(P_GAINS)
    trials = blocks * args.repeats * len(trial_steps(args.step_ticks))
    duration = trials * (args.hold_s + args.return_s) + 2 * blocks * args.return_s
    print(f"Arm joints: {', '.join(args.joints)}; P gains: {list(P_GAINS)}.\n"
          "I, D, startup force, integral limit, and both dead zones stay at their current values.\n"
          f"Steps in both directions: {args.step_ticks} ticks from each measured starting angle.\n"
          f"{trials} trials; hold={args.hold_s}s, center return={args.return_s}s, "
          f"feedback={args.sample_hz} Hz; approximately {duration / 60:.1f} minutes.\n"
          "Logs position, current, drive duty cycle, voltage, and individual read times.")
    if not args.execute:
        print("Preview only: no hardware connection. Add --execute when ready to move in clear air.")
        return
    print("Will hold all joints at their measured pose and test one arm joint at a time.\n"
          "Each joint's original P is restored before testing the next joint.\n"
          "Cleanup restores P and torque/lock states; initially unpowered joints relax.\n"
          f"Allow about +/-{(2 * max(args.step_ticks) + 3) * 360 / 4095:.2f} degrees of clear travel on each selected joint; no collision checker.")
    execute(args)


if __name__ == "__main__":
    main()
