#!/usr/bin/env python3
"""Compare pan acceleration=254 -> 0 -> 254 on the same 50 Hz trajectory.

Default: preview only, no drivers or robot connection.
From src, when ready: python compare_so101_streaming.py --execute

Uses the real LeRobot send_action(), but connects ONLY the bus: follower.connect()
would configure the servos and overwrite gains. Pan P=48, D=32, I=0, both dead
zones=0 and startup force=16 stay fixed across all three blocks. Only pan
acceleration (byte 41) varies between blocks; no multi-register profile writes.
Startup writes use byte 24 only; byte 25 is preserved. Goal velocity and factory
acceleration registers 85/86 are unchanged. Other joints and the gripper hold their
initial encoder positions. No IK or collision checker: start in clear air with
about +/-3 degrees of free pan travel for the default +/-2 degree comparison.
Cleanup holds the measured pan position and restores acceleration, gains, dead
zones, startup force, lock and torque states (also attempted after exceptions or Ctrl-C);
initially unpowered joints relax again. Do not run another bus client.

All blocks use the same rest-to-rest quintic trajectory (default +/-2 degrees,
2 seconds) and endpoint hold (2 seconds), repeated in both directions. The two
acceleration=254 blocks are kept separate to expose drift. Return-to-center moves
use acceleration=254 in every block to establish the same starting pose.
They are logged separately and excluded from comparisons. Sequential goal,
position, current, drive-output and voltage reads
perturb timing; each transaction is timestamped and missed read slots are logged.
The requested 50 Hz command rate is not a guarantee that the bus keeps up.
Smoothness summaries count encoder pauses while the goal advances, sample-to-
sample jumps, speed peaks and reverse travel in the central 60% of each stream.
These are quantized feedback measures, not a direct measurement of physical jerk;
always compare them alongside tracking error and completed travel.
Acceleration=0 is a firmware-dependent setting, not a verified profile-disable
switch: register 86 takes effect at zero and is logged but never changed.

--experiment duplicates compares full stream -> changed targets only -> full
stream, with acceleration=254 throughout. Changed targets keep their original
50 Hz deadlines; observation duration and all other settings stay identical.
--experiment speed uses 2 -> 1 -> 2 -> 4 -> 2 second streams, all with
acceleration=254 and full 50 Hz writes. Travel, gains and return motion stay fixed.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time


PAN = "shoulder_pan"
JOINTS = (PAN, "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
RATE_HZ = 50
ACCELERATIONS = (254, 0, 254)
RETURN_ACCELERATION = 254
FIXED_SETTINGS = dict(P_Coefficient=48, D_Coefficient=32, I_Coefficient=0, CW_Dead_Zone=0,
                      CCW_Dead_Zone=0, Startup_Byte_24=16)
TUNING_REGISTERS = (*FIXED_SETTINGS, "Acceleration")
DEG_PER_TICK = 360 / 4095
ROOT = Path(__file__).resolve().parent.parent
CALIBRATION = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/my_arm.json"
PROTECTED = (
    "P_Coefficient", "I_Coefficient", "D_Coefficient", "CW_Dead_Zone", "CCW_Dead_Zone",
    "Startup_Byte_24", "Byte_25", "Acceleration", "Goal_Velocity", "Torque_Limit",
    "Operating_Mode", "Angular_Resolution", "Phase", "Lock",
    "Maximum_Acceleration", "Acceleration_Multiplier_Raw",
)
SNAPSHOT = (*PROTECTED, "Firmware_Major_Version", "Firmware_Minor_Version",
            "Torque_Enable", "Goal_Position", "Present_Position")
COMMON = ("trial", "phase", "block", "repeat", "d_gain", "p_gain", "i_gain", "dead_zone",
          "startup_force", "rate_hz", "direction", "center_raw", "endpoint_raw",
          "pan_speed_raw", "pan_acceleration_raw", "profile_duration_s", "skip_duplicates")
TELEMETRY = {"current": "Present_Current", "drive_output": "Present_Load", "voltage": "Present_Voltage",
             "goal_echo": "Goal_Echo_Raw"}
COMMAND_FIELDS = (*COMMON, "sequence", "planned_s", "ideal_raw", "intended_raw", "action_deg",
                  "send_start_s", "send_end_s", "write_ms", "lateness_ms", "interval_ms", "status")
FEEDBACK_FIELDS = (*COMMON, "latest_sequence", "sample_planned_s", "intended_raw", "goal_raw", "actual_raw",
                   "goal_mismatch_ticks", "tracking_error_ticks", "servo_error_ticks", "tracking_error_deg",
                   "reference_raw", "reference_error_deg",
                   "goal_read_start_s", "goal_read_end_s", "position_read_start_s", "position_read_end_s",
                   *(f"{channel}_{suffix}" for channel in TELEMETRY for suffix in ("raw", "read_start_s", "read_end_s")),
                   "current_ma", "drive_output_pct", "voltage_v", "read_duration_ms",
                   "elapsed_s", "interval_ms", "skipped_read_slots")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--experiment", choices=("acceleration", "duplicates", "speed"), default="acceleration")
    parser.add_argument("--port", default="/dev/tty.usbmodem5B8E1155331")
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument("--travel-deg", type=float, default=2.0, help="pan travel from center, each direction (0.5-3 degrees)")
    parser.add_argument("--duration-s", type=float, default=2.0, help="quintic stream duration (1-5 seconds)")
    parser.add_argument("--hold-s", type=float, default=2.0, help="endpoint observation after the stream (1-5 seconds)")
    parser.add_argument("--read-hz", type=float, default=50.0, help="same feedback rate in every condition (10-100 Hz)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--return-timeout-s", type=float, default=5.0)
    parser.add_argument("--return-duration-s", type=float, default=0.,
                        help="optional quintic return to center, rather than a single target (0-2 seconds)")
    parser.add_argument("--start-tolerance-ticks", type=int, choices=(1, 2), default=1,
                        help="allowed measured start deviation; logged for matched-start analysis")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/so101_streaming")
    args = parser.parse_args(argv)
    for field, low, high in (("travel_deg", .5, 3), ("duration_s", 1, 5), ("hold_s", 1, 5),
                             ("read_hz", 10, 100), ("return_timeout_s", 1, 10), ("return_duration_s", 0, 2)):
        value = getattr(args, field)
        if not math.isfinite(value) or not low <= value <= high:
            parser.error(f"{field.replace('_', '-')} must be in [{low}, {high}]")
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.return_timeout_s <= args.return_duration_s + .25:
        parser.error("return-timeout-s must exceed return-duration-s by more than 0.25 seconds")
    return args


def profile_progress(elapsed, duration):
    """Quintic time scaling: zero velocity and acceleration at either endpoint."""
    u = max(0., min(1., elapsed / duration))
    return u**3 * (10 + u * (-15 + 6*u))


def command_schedule(center, endpoint, duration, skip_duplicates=False):
    """One identical 50 Hz path for every acceleration block, including the endpoint."""
    deadlines = [i/RATE_HZ for i in range(math.ceil(duration*RATE_HZ))]
    deadlines.append(duration)
    result = []
    for elapsed in deadlines:
        progress = profile_progress(elapsed, duration)
        ideal = center + (endpoint - center) * progress
        target = endpoint if elapsed == duration else round(ideal)
        if not skip_duplicates or not result or target != result[-1][2]:
            result.append((elapsed, ideal, target))
    return result


def trial_order(repeats, accelerations=ACCELERATIONS):
    for block, acceleration in enumerate(accelerations, 1):
        for repeat in range(1, repeats + 1):
            for direction in (1, -1):
                yield block, acceleration, repeat, direction


def action_for_raw(raw, calibration):
    # Select the middle of each integer's conversion bin to prevent floating-point
    # round trips from commanding the neighboring count, especially on held joints.
    action = {}
    for name, value in raw.items():
        cal = calibration[name]
        if name == "gripper":
            action[f"{name}.pos"] = min(100., (value + .5 - cal.range_min) * 100 / (cal.range_max - cal.range_min))
        else:
            action[f"{name}.pos"] = (value + .5 - (cal.range_min + cal.range_max) / 2) * DEG_PER_TICK
    return action


def summarize_smoothness(feedback, duration, direction, read_hz):
    """Coarse encoder measures during the central 60%, excluding sampling gaps.

    A pause needs >=100 ms at one encoder count while the read-back goal advances
    >=2 counts. Ordinary one-count quantization and the endpoint hold do not count.
    """
    rows = [r for r in feedback if .2 * duration <= r["elapsed_s"] <= .8 * duration]
    pairs, pauses = [], []
    run = rows[0] if rows else None
    gaps = 0

    def finish_pause(last):
        elapsed = last["elapsed_s"] - run["elapsed_s"]
        advance = direction * (last["goal_raw"] - run["goal_raw"])
        if elapsed >= .1 - 1e-9 and advance >= 2:
            pauses.append(elapsed)

    for a, b in zip(rows, rows[1:]):
        dt = b["elapsed_s"] - a["elapsed_s"]
        valid = 0 < dt <= 2.5 / read_hz
        if valid:
            pairs.append((b["actual_raw"] - a["actual_raw"], dt))
        else:
            gaps += 1
        if not valid or b["actual_raw"] != a["actual_raw"]:
            finish_pause(a)
            run = b
    if rows:
        finish_pause(rows[-1])
    return {
        "central_feedback_pairs": len(pairs), "central_sampling_gaps": gaps,
        "central_pause_count": len(pauses) if pairs else None,
        "central_pause_total_s": sum(pauses) if pairs else None,
        "central_longest_pause_s": max(pauses, default=0.) if pairs else None,
        "central_max_sample_jump_deg": max(abs(delta) * DEG_PER_TICK for delta, _ in pairs) if pairs else None,
        "central_max_sample_speed_deg_s": max(abs(delta) * DEG_PER_TICK / dt for delta, dt in pairs) if pairs else None,
        "central_reverse_travel_deg": sum(max(0, -direction * delta) * DEG_PER_TICK for delta, _ in pairs) if pairs else None,
    }


def summarize(commands, feedback, baseline, endpoint, direction, read_hz=50):
    delay = {}
    for threshold in (1, 2):
        delay[f"movement_delay_{threshold}_ticks_s"] = next((
            a["elapsed_s"] for a, b in zip(feedback, feedback[1:])
            if all(direction * (row["actual_raw"] - baseline) >= threshold for row in (a, b))
        ), None)
    for percent in (10, 50, 90):
        threshold = abs(endpoint-baseline) * percent / 100
        delay[f"time_to_{percent}_percent_s"] = next((
            a["elapsed_s"] for a, b in zip(feedback, feedback[1:])
            if all(direction * (row["actual_raw"] - baseline) >= threshold for row in (a, b))
        ), None)
    t10, t90 = delay["time_to_10_percent_s"], delay["time_to_90_percent_s"]
    delay["travel_10_to_90_s"] = None if t10 is None or t90 is None else t90-t10
    tail = [row for row in feedback if row["elapsed_s"] >= feedback[-1]["elapsed_s"] - .25]
    changed = next((row for row in commands if row["intended_raw"] != row["center_raw"]), None)
    intervals = [row["interval_ms"] for row in commands if row["interval_ms"] is not None]
    nominal = commands[0].get("profile_duration_s")
    motion = [row for row in feedback if nominal and row["elapsed_s"] <= nominal]
    return {
        **summarize_smoothness(feedback, nominal, direction, read_hz),
        **delay, "first_changed_target_s": changed["send_start_s"] if changed else None,
        "movement_delay_after_changed_target_s": (delay["movement_delay_1_ticks_s"] - changed["send_start_s"]
            if changed and delay["movement_delay_1_ticks_s"] is not None else None),
        "motion_mean_abs_tracking_error_deg": statistics.mean(abs(r["tracking_error_deg"]) for r in motion) if motion else None,
        "motion_rms_tracking_error_deg": math.sqrt(statistics.mean(r["tracking_error_deg"]**2 for r in motion)) if motion else None,
        "nominal_profile_mean_abs_error_deg": statistics.mean(abs(r["reference_error_deg"]) for r in motion) if motion else None,
        "tail_mean_abs_endpoint_error_deg": statistics.mean(abs(endpoint-r["actual_raw"]) * DEG_PER_TICK for r in tail),
        "completed_travel_fraction": direction * (statistics.mean(r["actual_raw"] for r in tail) - baseline) / abs(endpoint - baseline),
        "peak_endpoint_overshoot_deg": max(0, max(direction * (r["actual_raw"]-endpoint) for r in feedback)) * DEG_PER_TICK,
        "goal_mismatch_samples": sum(r["goal_mismatch_ticks"] != 0 for r in feedback),
        "feedback_samples": len(feedback), "commands_sent": len(commands),
        "duplicate_target_transitions": sum(a["intended_raw"] == b["intended_raw"] for a, b in zip(commands, commands[1:])),
        "max_command_lateness_ms": max(r["lateness_ms"] for r in commands),
        "max_write_ms": max(r["write_ms"] for r in commands),
        "median_command_interval_ms": statistics.median(intervals) if intervals else None,
        "max_feedback_interval_ms": max(r["interval_ms"] or 0 for r in feedback),
        "max_read_duration_ms": max(r["read_duration_ms"] for r in feedback),
        "skipped_read_slots": sum(r["skipped_read_slots"] for r in feedback),
        "peak_current_ma": max(r["current_ma"] for r in feedback),
        "mean_current_ma": statistics.mean(r["current_ma"] for r in feedback),
        "peak_abs_drive_output_pct": max(abs(r["drive_output_pct"]) for r in feedback),
        "min_voltage_v": min(r["voltage_v"] for r in feedback),
        "max_voltage_v": max(r["voltage_v"] for r in feedback),
    }


class Recorder:
    def __init__(self, arm, args, held, command_writer, feedback_writer):
        self.arm, self.bus, self.args = arm, arm.bus, args
        self.held = held
        self.center = held[PAN]
        self.travel = round(args.travel_deg / DEG_PER_TICK)
        self.envelope = self.travel + math.ceil(1 / DEG_PER_TICK)
        self.command_writer, self.feedback_writer = command_writer, feedback_writer

    def read(self, register, name=PAN):
        return int(self.bus.read(register, name, normalize=False, num_retry=0))

    def check(self, raw):
        cal = self.arm.calibration[PAN]
        if not cal.range_min <= raw <= cal.range_max or abs(raw - self.center) > self.envelope:
            raise RuntimeError(f"Pan position/goal {raw} is outside the calibrated test window")

    def run(self, metadata, schedule, window, settle=False):
        """One serial event loop; command deadlines match execute_trajectory()."""
        commands, feedback = [], []
        epoch = time.perf_counter()
        next_read, index, previous_send, previous_read = 0., 0, None, None
        stable_since = None
        while True:
            now = time.perf_counter() - epoch
            # As in execute_trajectory, retain overdue commands rather than hiding
            # missed deadlines. Log catch-up bursts; never issue concurrent bus I/O.
            if index < len(schedule) and now >= schedule[index][0]:
                planned, ideal, target = schedule[index]
                self.check(target)
                raw = {**self.held, PAN: target}
                action = action_for_raw(raw, self.arm.calibration)
                # Use the installed conversion, not just our assumed formula.
                encoded = self.bus._unnormalize({self.bus.motors[n].id: action[f"{n}.pos"] for n in JOINTS})
                if any(encoded[self.bus.motors[n].id] != raw[n] for n in JOINTS):
                    raise RuntimeError("LeRobot conversion does not match intended raw targets")
                begin = time.perf_counter() - epoch
                row = dict(metadata, sequence=index, planned_s=planned, ideal_raw=ideal,
                           intended_raw=target, action_deg=action[f"{PAN}.pos"], send_start_s=begin,
                           interval_ms=None if previous_send is None else 1000*(begin-previous_send),
                           lateness_ms=1000*(begin-planned), status="attempted")
                try:
                    self.arm.send_action(action)
                    row["status"] = "sent"
                finally:
                    row["send_end_s"] = time.perf_counter() - epoch
                    row["write_ms"] = 1000*(row["send_end_s"]-begin)
                    self.command_writer.writerow(row)
                commands.append(row)
                previous_send = begin
                index += 1
                continue
            if now >= next_read:
                row = dict(metadata, latest_sequence=index-1, sample_planned_s=next_read,
                           intended_raw=commands[-1]["intended_raw"])
                row["goal_read_start_s"] = time.perf_counter() - epoch
                row["goal_raw"] = self.read("Goal_Position")
                row["goal_read_end_s"] = time.perf_counter() - epoch
                row["position_read_start_s"] = time.perf_counter() - epoch
                row["actual_raw"] = self.read("Present_Position")
                row["position_read_end_s"] = time.perf_counter() - epoch
                for channel, register in TELEMETRY.items():
                    row[f"{channel}_read_start_s"] = time.perf_counter() - epoch
                    row[f"{channel}_raw"] = self.read(register)
                    row[f"{channel}_read_end_s"] = time.perf_counter() - epoch
                read_end = time.perf_counter() - epoch
                # LeRobot already sign-decodes Present_Load (direction bit 10).
                # This is motor PWM duty, not a measurement of joint torque.
                row.update(current_ma=row["current_raw"] * 6.5,
                           drive_output_pct=row["drive_output_raw"] * .1,
                           voltage_v=row["voltage_raw"] * .1,
                           read_duration_ms=1000*(read_end-row["goal_read_start_s"]))
                elapsed = row["position_read_end_s"]
                reference = (self.center + (metadata["endpoint_raw"]-self.center) * profile_progress(elapsed, metadata["profile_duration_s"])
                             if metadata["phase"] == "test" else row["intended_raw"])
                row.update(elapsed_s=elapsed, goal_mismatch_ticks=row["goal_raw"]-row["intended_raw"],
                           tracking_error_ticks=row["intended_raw"]-row["actual_raw"],
                           servo_error_ticks=row["goal_raw"]-row["actual_raw"],
                           tracking_error_deg=(row["intended_raw"]-row["actual_raw"])*DEG_PER_TICK,
                           reference_raw=reference, reference_error_deg=(reference-row["actual_raw"])*DEG_PER_TICK,
                           interval_ms=None if previous_read is None else 1000*(elapsed-previous_read))
                next_read += 1 / self.args.read_hz
                skipped = max(0, math.floor((read_end-next_read)*self.args.read_hz) + 1)
                row["skipped_read_slots"] = skipped
                next_read += skipped / self.args.read_hz  # No catch-up bursts of reads.
                self.feedback_writer.writerow(row)
                feedback.append(row)
                self.check(row["goal_raw"])
                self.check(row["actual_raw"])
                previous_read = elapsed
                if settle:
                    near = abs(row["actual_raw"]-self.center) <= self.args.start_tolerance_ticks and row["goal_raw"] == self.center
                    stable_since = (elapsed if stable_since is None else stable_since) if near else None
                    if stable_since is not None and elapsed - stable_since >= .25:
                        return commands, feedback
                    if elapsed >= window:
                        raise TimeoutError(f"Pan did not return to within {self.args.start_tolerance_ticks} tick(s) of center for 0.25 s")
                elif elapsed >= window and index == len(schedule):
                    return commands, feedback
                continue
            deadline = min(next_read, schedule[index][0] if index < len(schedule) else math.inf)
            time.sleep(max(0, epoch + deadline - time.perf_counter()))


def run_comparison(arm, args):
    bus = arm.bus
    duplicates = args.experiment == "duplicates"
    speed = args.experiment == "speed"
    durations = (2., 1., 2., 4., 2.) if speed else (args.duration_s,) * 3
    accelerations = (254,) * len(durations) if duplicates or speed else ACCELERATIONS
    total_trials = len(durations) * args.repeats * 2
    # Explicit byte 24 alias: never write the driver's ambiguous two-byte field.
    bus.model_ctrl_table = {name: dict(table) for name, table in bus.model_ctrl_table.items()}
    bus.model_ctrl_table["sts3215"].update({"Startup_Byte_24": (24, 1), "Byte_25": (25, 1),
                                          "Acceleration_Multiplier_Raw": (86, 1), "Goal_Echo_Raw": (67, 2)})
    layouts = {"P_Coefficient": (21, 1), "D_Coefficient": (22, 1), "I_Coefficient": (23, 1),
               "CW_Dead_Zone": (26, 1), "CCW_Dead_Zone": (27, 1), "Lock": (55, 1),
               "Acceleration": (41, 1), "Goal_Velocity": (46, 2), "Phase": (18, 1),
               "Maximum_Acceleration": (85, 1), "Present_Current": (69, 2),
               "Present_Load": (60, 2), "Present_Voltage": (62, 1)}
    for register, layout in layouts.items():
        if bus.model_ctrl_table["sts3215"].get(register) != layout:
            raise RuntimeError(f"Unexpected STS {register} register layout; no connection made")
    output = args.output_dir.expanduser() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    experiment = ("pan_streaming_duration_2_1_2_4_2" if speed else
                  "pan_streaming_duplicates_full_changed_full" if duplicates else "pan_streaming_acceleration_254_0_254")
    report = dict(experiment=experiment, schema_version=6,
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  rate_hz=RATE_HZ, accelerations_raw=accelerations, durations_s=durations,
                  return_acceleration_raw=RETURN_ACCELERATION, fixed_pan_settings=FIXED_SETTINGS,
                  deg_per_tick=DEG_PER_TICK, status="starting", trials=[], restoration_errors=[],
                  comparison_note="Three sequential pan acceleration=254/0/254 blocks, writing only byte 41 for acceleration. Same rounded quintic targets at 50 Hz, center/endpoints and duration. Each trial starts within one encoder tick of center. Returns use acceleration=254 and are excluded from comparisons. Pan P=48, D=32, I=0, dead zones=0, startup=16 throughout; goal velocity, registers 85/86 and other joints' settings remain original.",
                  acceleration_note="Acceleration=0 is not a verified profile-disable switch. The factory acceleration multiplier (byte 86) takes effect at zero; it is logged but unchanged.",
                  smoothness_note="Central 60% of the nominal stream only: pauses are >=0.10 s at one encoder count while the read-back goal advances >=2 counts in the commanded direction. Gaps >2.5 requested read periods split pauses and are excluded from sample jump/speed/reverse-travel metrics. These quantized observations are not physical jerk; compare with tracking, completed travel and timing. Missing valid pairs produce null metrics.",
                  timing_note="Reads are sequential and perturb timing; goal readback is sampled, not acknowledgment of every command. Overdue commands are retained; missed read slots are skipped. Movement delay is from trajectory start, with a separate delay from the first changed integer target. Serial transaction times bound measurement precision.",
                  target_note="intended_raw is the last transmitted integer goal; reference_raw is the ideal continuous quintic at the position timestamp. Return moves are excluded from trial summaries.",
                  feedback_units={"drive_output_raw": "LeRobot sign-decoded Present_Load; 0.1% PWM duty/count, not measured joint torque",
                                  "current_raw": "Present_Current; nominal 6.5 mA/count",
                                  "voltage_raw": "Present_Voltage; 0.1 V/count",
                                  "goal_echo_raw": "Read-only goal position at 67 (two bytes). Its relationship to the internal trajectory is firmware-dependent; do not assume this is the profile reference."},
                  profile_source="https://ftservo.github.io/ftservo-wiki/reference/parameter/memory-sts/",
                  settings_changes=[])
    if duplicates:
        report["comparison_note"] = (
            "Full stream / changed integer targets only / full stream. All pan settings fixed: "
            "P=48, D=32, I=0, dead zones=0, startup=16, acceleration=254. The middle block "
            "omits writes when the integer target is unchanged; retained deadlines are identical. "
            "Feedback remains at the same rate and observation lasts duration + hold in all blocks. "
            "Returns are excluded; each trial starts within one encoder tick of the shared center.")
    if speed:
        report["comparison_note"] = (
            "2 / 1 / 2 / 4 / 2 second quintic streams, fixed travel, full 50 Hz writes, "
            "P=48, D=32, I=0, dead zones=0, startup=16, acceleration=254 throughout. "
            "Return trajectory and endpoint hold duration are constant. Baseline duration repeats "
            "between test durations to expose drift. Each start is recorded; compare matching start "
            "offsets and directions. Raw pause duration must be interpreted relative to stream duration.")
    report["comparison_note"] = report["comparison_note"].replace(
        "one encoder tick", f"{args.start_tolerance_ticks} encoder tick(s)")
    report["start_note"] = "Record actual starts and compare matching start offsets; do not attribute unequal starting errors to the treatment."
    original, recorder, changed = {}, None, False

    def read(register, name):
        return int(bus.read(register, name, normalize=False, num_retry=0))

    def write(register, name, value):
        if register not in ("Goal_Position", "Torque_Enable") and not (name == PAN and register in (*TUNING_REGISTERS, "Lock")):
            raise ValueError(f"Diagnostic cannot write {register}")
        event = None
        if register in (*TUNING_REGISTERS, "Lock"):
            event = dict(register=register, motor=name, value=int(value), status="attempted")
            report["settings_changes"].append(event)
            save()
        bus.write(register, name, int(value), normalize=False)
        if read(register, name) != int(value):
            raise RuntimeError(f"{name} {register} readback failed")
        if event is not None:
            event["status"] = "verified"
            save()

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def verify_settings(pan_settings=None):
        for name in JOINTS:
            for register in PROTECTED:
                value = read(register, name)
                expected = (pan_settings[register] if name == PAN and pan_settings and register in pan_settings
                            else original[name][register])
                if value != expected:
                    raise RuntimeError(f"{name} {register}: expected {expected}, read {value}")

    def set_pan_register(register, value):
        if register == "Acceleration":
            # SRAM: verified (41, 1) above. Do not unlock EEPROM or touch the
            # neighboring goal-position bytes, speed, or factory profile settings.
            write(register, PAN, value)
            return
        # Match the existing gain diagnostic: unlock, write/read back, relock.
        # Keep torque on and hold the current pose while changing parameters.
        write("Lock", PAN, 0)
        try:
            write(register, PAN, value)
        finally:
            write("Lock", PAN, 1)

    def set_pan_settings(settings):
        for register in TUNING_REGISTERS:
            if read(register, PAN) != settings[register]:
                set_pan_register(register, settings[register])
        if read("Lock", PAN) != 1:
            write("Lock", PAN, 1)

    def hold_pan():
        actual = read("Present_Position", PAN)
        cal = arm.calibration[PAN]
        if not cal.range_min <= actual <= cal.range_max:
            raise RuntimeError("Measured pan position is outside calibrated limits")
        write("Goal_Position", PAN, actual)

    def return_to_center(metadata):
        schedule = (command_schedule(read("Present_Position", PAN), recorder.center, args.return_duration_s)
                    if args.return_duration_s else [(0., float(recorder.center), recorder.center)])
        recorder.run(metadata, schedule, args.return_timeout_s, settle=True)

    try:
        bus.connect()  # No follower.connect(), configure(), or calibration writes.
        if not bus.is_calibrated or set(arm.calibration) != set(JOINTS):
            raise RuntimeError("Saved calibration does not match the motors; no settings changed")
        original = {name: {reg: read(reg, name) for reg in SNAPSHOT} for name in JOINTS}
        report["original_registers"] = original
        for name, state in original.items():
            cal = arm.calibration[name]
            if state["Operating_Mode"] != 0 or state["Torque_Enable"] not in (0, 1):
                raise RuntimeError("Expected position mode and torque state 0 or 1")
            if cal.drive_mode != 0 or cal.id != bus.motors[name].id:
                raise RuntimeError("Expected matching motor IDs and non-inverted calibration")
            if not cal.range_min < cal.range_max or not cal.range_min <= state["Present_Position"] <= cal.range_max:
                raise RuntimeError(f"{name} is outside its calibrated range")
        if original[PAN]["Angular_Resolution"] != 1 or bus.model_resolution_table["sts3215"] != 4096:
            raise RuntimeError("Expected a 4096-count pan encoder at angular resolution 1")
        held = {name: state["Present_Position"] for name, state in original.items()}
        with (output / "commands.csv").open("w", newline="") as commands_file, \
                (output / "feedback.csv").open("w", newline="") as feedback_file:
            cw, fw = csv.DictWriter(commands_file, COMMAND_FIELDS), csv.DictWriter(feedback_file, FEEDBACK_FIELDS)
            cw.writeheader()
            fw.writeheader()
            recorder = Recorder(arm, args, held, cw, fw)
            recorder.check(recorder.center - recorder.envelope)
            recorder.check(recorder.center + recorder.envelope)
            report.update(center_raw=recorder.center, travel_ticks=recorder.travel,
                          actual_travel_deg=recorder.travel*DEG_PER_TICK, envelope_ticks=recorder.envelope,
                          nominal_profile=dict(type="quintic", durations_s=durations))
            save()  # Original state is durable before the first motor write.
            changed = True
            for name in JOINTS:
                write("Goal_Position", name, held[name])
            for name in JOINTS:
                if original[name]["Torque_Enable"] == 0:
                    write("Torque_Enable", name, 1)
            return_settings = {**FIXED_SETTINGS, "Acceleration": RETURN_ACCELERATION, "Lock": 1}
            for trial, (block, acceleration, repeat, direction) in enumerate(trial_order(args.repeats, accelerations), 1):
                duration = durations[block-1]
                endpoint = recorder.center + direction * recorder.travel
                metadata = dict(trial=trial, phase="return", block=block, repeat=repeat,
                                d_gain=32, p_gain=48, i_gain=0, dead_zone=0, startup_force=16, rate_hz=RATE_HZ,
                                direction=direction, center_raw=recorder.center, endpoint_raw=endpoint,
                                pan_speed_raw=original[PAN]["Goal_Velocity"],
                                pan_acceleration_raw=RETURN_ACCELERATION, profile_duration_s=duration,
                                skip_duplicates=False)
                report["active_trial"] = metadata.copy()
                save()
                try:
                    # Return with the same settings before every measured trial.
                    hold_pan()
                    set_pan_settings(return_settings)
                    verify_settings(return_settings)
                    return_to_center(metadata)
                    settings = {**FIXED_SETTINGS, "Acceleration": acceleration, "Lock": 1}
                    set_pan_settings(settings)
                    verify_settings(settings)
                    start_positions = {name: read("Present_Position", name) for name in JOINTS}
                    if abs(start_positions[PAN]-recorder.center) > args.start_tolerance_ticks:
                        raise RuntimeError("Pan moved away from center before the trial")
                    if any(abs(start_positions[n]-held[n]) * DEG_PER_TICK > 2 for n in JOINTS if n != PAN):
                        raise RuntimeError("A held joint moved more than two degrees from its starting pose")
                    metadata.update(phase="test", pan_acceleration_raw=acceleration,
                                    skip_duplicates=duplicates and block == 2)
                    report["active_trial"] = {**metadata, "start_positions_raw": start_positions}
                    save()
                    commands, feedback = recorder.run(metadata,
                        command_schedule(recorder.center, endpoint, duration, metadata["skip_duplicates"]), duration + args.hold_s)
                    verify_settings(settings)
                    report["trials"].append({**metadata, "start_positions_raw": start_positions,
                                             **summarize(commands, feedback, start_positions[PAN], endpoint, direction, args.read_hz)})
                    report.pop("active_trial")
                    save()
                    print(f"Trial {trial}/{total_trials}: block {block}, duration={duration}s, acceleration={acceleration}, skip_duplicates={metadata['skip_duplicates']}, P=48, D=32, direction {direction:+d}", flush=True)
                finally:
                    commands_file.flush()
                    feedback_file.flush()
            # End at the shared center as well as returning there between trials.
            hold_pan()
            set_pan_settings(return_settings)
            verify_settings(return_settings)
            return_to_center({**metadata, "phase": "final_return", "skip_duplicates": False,
                              "pan_acceleration_raw": RETURN_ACCELERATION})
            report["block_comparison"] = []
            metrics = ("movement_delay_after_changed_target_s", "travel_10_to_90_s",
                       "motion_mean_abs_tracking_error_deg", "tail_mean_abs_endpoint_error_deg",
                       "peak_endpoint_overshoot_deg", "completed_travel_fraction",
                       "central_pause_count", "central_pause_total_s", "central_longest_pause_s",
                       "central_max_sample_jump_deg", "central_max_sample_speed_deg_s", "central_reverse_travel_deg",
                       "peak_current_ma", "peak_abs_drive_output_pct", "min_voltage_v")
            for block, acceleration in enumerate(accelerations, 1):
                for direction in (1, -1):
                    rows = [t for t in report["trials"] if t["direction"] == direction and t["block"] == block]
                    medians = {}
                    for metric in metrics:
                        values = [t[metric] for t in rows if t[metric] is not None]
                        medians[f"median_{metric}"] = statistics.median(values) if values else None
                        medians[f"observations_{metric}"] = len(values)
                    report["block_comparison"].append(dict(block=block, pan_acceleration_raw=acceleration,
                                                           profile_duration_s=durations[block-1],
                                                           skip_duplicates=duplicates and block == 2,
                                                           p_gain=48, d_gain=32, direction=direction, **medians))
            report["status"] = "complete"
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if changed and not bus.is_connected:
            report["restoration_errors"].append("Bus disconnected before cleanup")
        if changed and bus.is_connected:
            def restore(label, action):
                try:
                    action()
                    return True
                except Exception as exc:
                    report["restoration_errors"].append(f"{label}: {exc}")
                    return False

            pan_ready = restore("hold current pan", hold_pan)
            if not pan_ready:
                restore("disable pan before settings restoration", lambda: write("Torque_Enable", PAN, 0))
            settings_ready = True
            for register in TUNING_REGISTERS:
                restored = restore(f"restore pan {register}",
                                   lambda register=register: set_pan_register(register, original[PAN][register]))
                settings_ready = restored and settings_ready
            lock_ready = restore("restore pan Lock", lambda: write("Lock", PAN, original[PAN]["Lock"]))
            for name in JOINTS:
                torque = original[name]["Torque_Enable"] if name != PAN or (pan_ready and settings_ready and lock_ready) else 0
                restore(f"restore {name} torque", lambda name=name, torque=torque: write("Torque_Enable", name, torque))
            restore("verify restored settings", verify_settings)
            report["final_registers"] = {}
            for name in JOINTS:
                def snapshot(name=name):
                    report["final_registers"][name] = {reg: read(reg, name) for reg in (*PROTECTED, "Torque_Enable")}
                    for reg, value in report["final_registers"][name].items():
                        if value != original[name][reg]:
                            raise RuntimeError(f"{name} {reg} was not restored: {value} != {original[name][reg]}")
                restore(f"read final {name} settings", snapshot)
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
            raise RuntimeError("Cleanup could not verify all original settings; see report.json")


def execute(args):
    import numpy  # noqa: F401 -- load before torch/LeRobot to avoid duplicate macOS libomp
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    calibration = args.calibration.expanduser().resolve(strict=True)
    arm = SO101Follower(SO101FollowerConfig(port=args.port, id=calibration.stem,
                       calibration_dir=calibration.parent, use_degrees=True, cameras={},
                       max_relative_target=None, disable_torque_on_disconnect=False))
    run_comparison(arm, args)


def main(argv=None):
    args = arguments(argv)
    ticks = round(args.travel_deg / DEG_PER_TICK)
    comparison = ("Pan full stream -> changed targets only -> full stream; acceleration fixed at 254"
                  if args.experiment == "duplicates" else "Pan acceleration=254 -> 0 -> 254 (byte 41 only)")
    if args.experiment == "speed":
        comparison = "Pan durations 2 -> 1 -> 2 -> 4 -> 2 seconds; acceleration fixed at 254"
    count = args.repeats * (10 if args.experiment == "speed" else 6)
    profile = "2/1/2/4/2s by block" if args.experiment == "speed" else f"{args.duration_s}s"
    print(f"{comparison} at 50 Hz; {count} trials, both directions.\n"
          f"Travel: {ticks} counts ({ticks*DEG_PER_TICK:.3f} degrees) from one shared center.\n"
          f"Quintic trajectory: {profile} plus {args.hold_s}s endpoint hold.\n"
          "Pan P=48, D=32, I=0, CW/CCW dead zones=0, startup force=16 (byte 24 only).\n"
          "Returns use acceleration=254 and are excluded from the comparison.\n"
          f"Goal, position, current, drive output and voltage readback: requested {args.read_hz} Hz.\n"
          "Goal velocity, registers 85/86 and other joints' settings stay unchanged; cleanup restores original settings.\n"
          "Acceleration=0 is not a verified profile-disable switch; the multiplier in byte 86 remains unchanged.\n"
          "Compares tracking, starting delay, pauses, sample jumps and overshoot for each block/direction.\n"
          "Logs commands.csv, feedback.csv, and report.json in outputs/so101_streaming.")
    if not args.execute:
        print("Preview only: no drivers loaded and no robot connection. --execute enables motion.")
        return
    print(f"Allow +/-{(ticks+math.ceil(1/DEG_PER_TICK))*DEG_PER_TICK:.2f} degrees of clear pan travel. "
          "No collision checker; cleanup restores torque states, so initially unpowered joints relax.")
    execute(args)


if __name__ == "__main__":
    main()
