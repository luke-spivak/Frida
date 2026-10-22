"""P-sweep routing and cleanup using a fake bus and clock, never motor drivers."""

from copy import deepcopy
import csv
import json
import sys
from types import SimpleNamespace

import pytest

import compare_so101_p_gains as diagnostic


class FakeBus:
    model_ctrl_table = {"sts3215": {diagnostic.STARTUP_REGISTER: (24, 2)}}
    model_resolution_table = {"sts3215": 4096}
    is_calibrated = True

    def __init__(self, clock):
        self.clock = clock
        self.is_connected = False
        self.writes = []
        self.fail = None
        self.registers = {}
        for i, name in enumerate(diagnostic.JOINTS):
            self.registers[name] = dict.fromkeys(diagnostic.REGISTERS, 0)
            self.registers[name].update({
                "P_Coefficient": 31 + i, "I_Coefficient": i, "D_Coefficient": 30 + i,
                "CW_Dead_Zone": 1, "CCW_Dead_Zone": 2,
                diagnostic.STARTUP_REGISTER: 16 + i, diagnostic.ADJACENT_REGISTER: 9 + i,
                "Angular_Resolution": 1, "Present_Position": 2000 + 100 * i,
                "Goal_Position": 2000 + 100 * i, "Torque_Enable": i % 2, "Lock": i % 2,
                "Present_Load": -20, "Present_Current": 3, "Present_Voltage": 54,
            })
        self.original = deepcopy(self.registers)

    def connect(self):
        self.is_connected = True

    def disconnect(self, disable_torque):
        assert disable_torque is False
        self.is_connected = False

    def read(self, register, joint, **kwargs):
        assert self.is_connected
        self.clock.now += .001
        return self.registers[joint][register]

    def write(self, register, joint, value, **kwargs):
        assert self.is_connected
        # These are the only permitted writes; notably no I/D/startup/dead zones.
        assert register in {"P_Coefficient", "Goal_Position", "Torque_Enable", "Lock"}
        state = self.registers[joint]
        if register == "P_Coefficient":
            assert joint != "gripper"
            assert state["Torque_Enable"] == 1  # No unloading gravity-bearing joints.
            assert state["Lock"] == 0
            if value != self.original[joint][register]:
                assert all(self.registers[name][register] == self.original[name][register]
                           for name in diagnostic.JOINTS if name != joint)
        self.writes.append((register, joint, value))
        state[register] = value
        if register == "Goal_Position":
            state["Present_Position"] = value
        self.clock.now += .001
        if self.fail:
            self.fail(register, joint, value)

    def enable_torque(self, joint):
        self.write("Torque_Enable", joint, 1)
        self.write("Lock", joint, 1)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    # Tests never import the LeRobot/serial hardware path, even accidentally.
    monkeypatch.setitem(sys.modules, "lerobot", None)
    clock = SimpleNamespace(now=0.)

    def sleep(seconds):
        clock.now += seconds

    monkeypatch.setattr(diagnostic, "time", SimpleNamespace(perf_counter=lambda: clock.now, sleep=sleep))
    args = diagnostic.arguments(["--output-dir", str(tmp_path), "--repeats", "1",
                                 "--hold-s", "2", "--return-s", ".5", "--sample-hz", "5"])
    bus = FakeBus(clock)
    calibration = {name: SimpleNamespace(range_min=0, range_max=4095) for name in diagnostic.JOINTS}
    return args, bus, calibration


def read_report(args):
    return json.loads(next(args.output_dir.glob("*/report.json")).read_text())


def assert_restored(bus):
    for joint in diagnostic.JOINTS:
        for register in ("P_Coefficient", "Torque_Enable", "Lock", *diagnostic.UNCHANGED_REGISTERS):
            assert bus.registers[joint][register] == bus.original[joint][register]
    assert not bus.is_connected


def test_all_joints_p_only_logging_and_restoration(rig):
    args, bus, calibration = rig
    diagnostic.run_comparison(args, bus, calibration)
    report = read_report(args)
    assert report["status"] == "complete"
    assert report["restoration_errors"] == []
    assert [(b["joint"], b["p_gain"]) for b in report["blocks"]] == [
        (joint, p) for joint in diagnostic.ARM_JOINTS for p in (32, 40, 48, 56)]
    assert_restored(bus)
    assert FakeBus.model_ctrl_table["sts3215"][diagnostic.STARTUP_REGISTER] == (24, 2)
    with next(args.output_dir.glob("*/samples.csv")).open() as stream:
        samples = list(csv.DictReader(stream))
    assert {r["joint"] for r in samples} == set(diagnostic.ARM_JOINTS)
    for row in samples:
        original = bus.original[row["joint"]]
        assert int(row["i_gain"]) == original["I_Coefficient"]
        assert int(row["startup_force"]) == original[diagnostic.STARTUP_REGISTER]
        assert float(row["current_ma"]) == 19.5
        assert float(row["drive_output_pct"]) == -2
        assert int(row["initial_error_ticks"]) == int(row["target_raw"]) - int(row["baseline_raw"])
    steps = [r for r in report["commands"] if r["phase"] == "step"]
    assert len(steps) == 5 * 4 * 6
    assert all(r["movement_delay_1_ticks_s"] is not None for r in steps)
    assert all(r["peak_overshoot_deg"] == 0 for r in steps)
    assert all(r["tail_mean_abs_error_deg"] == 0 for r in steps)
    gripper_goals = [value for register, joint, value in bus.writes
                     if register == "Goal_Position" and joint == "gripper"]
    assert set(gripper_goals) == {bus.original["gripper"]["Present_Position"]}


@pytest.mark.parametrize("error", [ConnectionError, KeyboardInterrupt])
def test_cleanup_after_partial_write_failure_on_later_joint(rig, error):
    args, bus, calibration = rig

    def fail(register, joint, value):
        if register == "P_Coefficient" and joint == "elbow_flex" and value == 48:
            bus.fail = None
            raise error("injected after the setting was applied")

    bus.fail = fail
    with pytest.raises(error):
        diagnostic.run_comparison(args, bus, calibration)
    assert_restored(bus)
    report = read_report(args)
    assert report["status"] == ("interrupted" if error is KeyboardInterrupt else "failed")
    assert report["restoration_errors"] == []


def test_selected_joint_only(rig):
    args, bus, calibration = rig
    args.joints = ["wrist_flex"]
    diagnostic.run_comparison(args, bus, calibration)
    assert {joint for register, joint, _ in bus.writes if register == "P_Coefficient"} == {"wrist_flex"}
    assert_restored(bus)


def test_all_selected_joint_limits_checked_before_any_write(rig):
    args, bus, calibration = rig
    bus.registers["wrist_roll"]["Present_Position"] = 4094
    with pytest.raises(RuntimeError, match="wrist_roll.*outside calibrated limits"):
        diagnostic.run_comparison(args, bus, calibration)
    assert bus.writes == []
    assert not bus.is_connected


def test_preview_does_not_enter_hardware_path(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "lerobot", None)
    monkeypatch.setattr(sys, "argv", ["compare_so101_p_gains.py"])
    diagnostic.main()
    assert "Preview only" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["--joints", "gripper"], ["--joints", "wrist_roll", "wrist_roll"]])
def test_reject_invalid_joint_selection(argv):
    with pytest.raises(SystemExit):
        diagnostic.arguments(argv)
