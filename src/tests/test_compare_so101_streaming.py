"""Offline checks: simulated bus/clock only; never construct a hardware follower."""

from copy import deepcopy
import csv
import io
import json
import sys
from types import SimpleNamespace

import pytest

import compare_so101_streaming as diagnostic


class FakeBus:
    model_ctrl_table = {"sts3215": {"Minimum_Startup_Force": (24, 2), "Phase": (18, 1),
                                   "Acceleration": (41, 1), "Goal_Velocity": (46, 2),
                                   "P_Coefficient": (21, 1), "D_Coefficient": (22, 1),
                                   "I_Coefficient": (23, 1), "CW_Dead_Zone": (26, 1),
                                   "CCW_Dead_Zone": (27, 1), "Lock": (55, 1),
                                   "Maximum_Acceleration": (85, 1), "Present_Current": (69, 2),
                                   "Present_Load": (60, 2), "Present_Voltage": (62, 1)}}
    model_resolution_table = {"sts3215": 4096}
    is_calibrated = True

    def __init__(self, clock):
        self.clock = clock
        self.is_connected = False
        self.writes = []
        self.motors = {name: SimpleNamespace(id=i) for i, name in enumerate(diagnostic.JOINTS, 1)}
        self.calibration = {name: SimpleNamespace(id=i, range_min=100, range_max=4000, drive_mode=0)
                            for i, name in enumerate(diagnostic.JOINTS, 1)}
        self.registers = {name: dict.fromkeys(diagnostic.SNAPSHOT, 0) for name in diagnostic.JOINTS}
        for i, state in enumerate(self.registers.values()):
            state.update(P_Coefficient=32+i, I_Coefficient=i, D_Coefficient=25+i,
                         CW_Dead_Zone=1, CCW_Dead_Zone=2, Startup_Byte_24=16+i, Byte_25=i,
                         Acceleration=140+i, Goal_Velocity=200+i, Phase=12,
                         Angular_Resolution=1, Torque_Enable=i % 2, Lock=i % 2,
                         Goal_Position=2000, Goal_Echo_Raw=2000, Present_Position=2000, Present_Current=7,
                         Present_Load=-150, Present_Voltage=54,
                         Maximum_Acceleration=254, Acceleration_Multiplier_Raw=2)
        self.original = deepcopy(self.registers)

    def connect(self):
        self.is_connected = True

    def disconnect(self, disable_torque):
        assert disable_torque is False
        self.is_connected = False

    def read(self, register, name, **kwargs):
        assert self.is_connected
        self.clock.now += .0005
        return self.registers[name][register]

    def write(self, register, name, value, **kwargs):
        assert self.is_connected
        assert register in {"Goal_Position", "Torque_Enable"} or (name == diagnostic.PAN and register in (*diagnostic.TUNING_REGISTERS, "Lock"))
        if register in diagnostic.FIXED_SETTINGS:
            assert self.registers[name]["Lock"] == 0
        if register == "Startup_Byte_24":
            assert self.model_ctrl_table["sts3215"][register] == (24, 1)
        if register == "Acceleration":
            assert name == diagnostic.PAN
            assert self.model_ctrl_table["sts3215"][register] == (41, 1)
        self.writes.append((register, name, value))
        self.registers[name][register] = value
        if register == "Goal_Position":
            self.registers[name]["Present_Position"] = value
            self.registers[name]["Goal_Echo_Raw"] = value
        self.clock.now += .0005

    def _unnormalize(self, values):
        result = {}
        for name, motor in self.motors.items():
            cal = self.calibration[name]
            value = values[motor.id]
            if name == "gripper":
                raw = int(min(100, max(0, value)) / 100 * (cal.range_max-cal.range_min) + cal.range_min)
            else:
                raw = int(value * 4095 / 360 + (cal.range_min+cal.range_max)/2)
            result[motor.id] = raw
        return result


class FakeArm:
    def __init__(self, clock):
        self.clock = clock
        self.bus = FakeBus(clock)
        self.calibration = self.bus.calibration
        self.sent = []
        self.sent_profiles = []
        self.send_hook = None
        self.drop = False

    def send_action(self, action):
        raw = self.bus._unnormalize({m.id: action[f"{n}.pos"] for n, m in self.bus.motors.items()})
        self.sent.append((self.clock.now, raw))
        self.sent_profiles.append(deepcopy(self.bus.registers))
        if not self.drop:
            for name, motor in self.bus.motors.items():
                self.bus.registers[name]["Goal_Position"] = raw[motor.id]
                self.bus.registers[name]["Goal_Echo_Raw"] = raw[motor.id]
                self.bus.registers[name]["Present_Position"] = raw[motor.id]
        self.clock.now += .0005
        if self.send_hook:
            self.send_hook()
        return action


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "lerobot", None)
    clock = SimpleNamespace(now=0.)

    def sleep(seconds):
        # Real perf_counter advances even after a zero-duration sleep; avoid a
        # floating-point deadline tie freezing the simulated event loop.
        clock.now += max(seconds, 1e-9)

    monkeypatch.setattr(diagnostic, "time", SimpleNamespace(perf_counter=lambda: clock.now, sleep=sleep))
    args = diagnostic.arguments(["--output-dir", str(tmp_path), "--repeats", "1"])
    arm = FakeArm(clock)
    return args, arm, clock


def recorder_for(arm, args):
    arm.bus.connect()
    buffers = io.StringIO(), io.StringIO()
    writers = [csv.DictWriter(stream, fields) for stream, fields in zip(
        buffers, (diagnostic.COMMAND_FIELDS, diagnostic.FEEDBACK_FIELDS))]
    for writer in writers:
        writer.writeheader()
    recorder = diagnostic.Recorder(arm, args, dict.fromkeys(diagnostic.JOINTS, 2000), *writers)
    metadata = dict(trial=1, phase="test", block=1, repeat=1, d_gain=32, rate_hz=50,
                    direction=1, center_raw=2000, endpoint_raw=2023,
                    profile_duration_s=args.duration_s)
    return recorder, metadata, buffers


def assert_restored(arm):
    for name in diagnostic.JOINTS:
        for register in (*diagnostic.PROTECTED, "Torque_Enable"):
            assert arm.bus.registers[name][register] == arm.bus.original[name][register]
    assert not arm.bus.is_connected


def test_quintic_schedule_is_symmetric_bounded_and_has_exact_endpoints():
    forward = diagnostic.command_schedule(2000, 2023, 2)
    reverse = diagnostic.command_schedule(2000, 1977, 2)
    assert len(forward) == len(reverse) == 101
    assert forward[0] == (0., 2000., 2000)
    assert forward[-1] == (2., 2023., 2023)
    assert reverse[-1] == (2., 1977., 1977)
    assert all(a[0] < b[0] and a[2] <= b[2] for a, b in zip(forward, forward[1:]))
    assert all(a[2]+b[2] == 4000 for a, b in zip(forward, reverse))
    assert diagnostic.profile_progress(1, 2) == .5
    assert diagnostic.profile_progress(-1, 2) == 0
    assert diagnostic.profile_progress(3, 2) == 1
    # Smooth starts/ends: the first/last 1% advances under 0.001% of travel.
    assert diagnostic.profile_progress(.02, 2) < .00001
    assert 1-diagnostic.profile_progress(1.98, 2) < .00001


def test_command_counts_readback_and_observation_window(rig):
    args, arm, _ = rig
    recorder, metadata, buffers = recorder_for(arm, args)
    commands, feedback = recorder.run(metadata, diagnostic.command_schedule(2000, 2023, 2), 4)
    assert len(commands) == 101
    assert all(r['goal_mismatch_ticks'] == r['servo_error_ticks'] == 0 for r in feedback)
    assert feedback[-1]['elapsed_s'] == pytest.approx(4, abs=.021)
    assert len(feedback) == 201
    assert feedback[-1]['actual_raw'] == 2023
    assert all(all(raw[m.id] == 2000 for n, m in arm.bus.motors.items() if n != diagnostic.PAN)
               for _, raw in arm.sent)
    assert len(list(csv.DictReader(io.StringIO(buffers[0].getvalue())))) == 101
    for row in feedback:
        assert row['current_raw'] == 7 and row['current_ma'] == 45.5
        assert row['drive_output_raw'] == -150 and row['drive_output_pct'] == -15
        assert row['voltage_raw'] == 54 and row['voltage_v'] == pytest.approx(5.4)
        assert row['position_read_end_s'] < row['current_read_start_s'] + 1e-9
        assert row['current_read_end_s'] < row['drive_output_read_end_s'] < row['voltage_read_end_s']
        assert row['read_duration_ms'] == pytest.approx(3.0)
        assert row['goal_echo_raw'] == row['goal_raw']
        assert row['goal_echo_read_end_s'] > row['voltage_read_end_s']
        assert row['elapsed_s'] == row['position_read_end_s']
    summary = diagnostic.summarize(commands, feedback, 2000, 2023, 1)
    assert summary['motion_mean_abs_tracking_error_deg'] == 0
    assert summary['peak_endpoint_overshoot_deg'] == 0
    assert summary['peak_current_ma'] == 45.5
    assert summary['peak_abs_drive_output_pct'] == 15
    assert summary['min_voltage_v'] == pytest.approx(5.4)
    assert summary['completed_travel_fraction'] == 1
    assert summary['central_pause_count'] == 0
    assert summary['central_reverse_travel_deg'] == 0


@pytest.mark.parametrize('direction', [1, -1])
def test_duplicate_suppression_preserves_every_change_and_its_deadline(direction):
    full = diagnostic.command_schedule(2000, 2000 + direction*23, 2)
    expected = [row for i, row in enumerate(full) if i == 0 or row[2] != full[i-1][2]]
    changed = diagnostic.command_schedule(2000, 2000 + direction*23, 2, True)
    assert changed == expected
    assert len(changed) == 24
    assert changed[-1][2] == full[-1][2]


def test_duplicate_comparison_keeps_settings_and_observation_window_and_restores(rig):
    args, arm, _ = rig
    args.experiment = 'duplicates'
    diagnostic.run_comparison(arm, args)
    report = json.loads(next(args.output_dir.glob('*/report.json')).read_text())
    assert report['status'] == 'complete'
    assert report['experiment'] == 'pan_streaming_duplicates_full_changed_full'
    assert report['accelerations_raw'] == [254, 254, 254]
    assert [t['commands_sent'] for t in report['trials']] == [101, 101, 24, 24, 101, 101]
    for t in report['trials']:
        assert t['feedback_samples'] == 201
        assert t['completed_travel_fraction'] == 1
        assert t['goal_mismatch_samples'] == 0
        assert t['skip_duplicates'] == (t['block'] == 2)
    for state in arm.sent_profiles:
        assert state[diagnostic.PAN]['Acceleration'] == 254
        for register, expected in diagnostic.FIXED_SETTINGS.items():
            assert state[diagnostic.PAN][register] == expected
    assert_restored(arm)


def test_duplicate_trial_interruption_restores_settings(rig):
    args, arm, _ = rig
    args.experiment = 'duplicates'
    def fail():
        if len(arm.sent) == 210:
            raise KeyboardInterrupt()
    arm.send_hook = fail
    with pytest.raises(KeyboardInterrupt):
        diagnostic.run_comparison(arm, args)
    report = json.loads(next(args.output_dir.glob('*/report.json')).read_text())
    assert report['active_trial']['skip_duplicates'] is True
    assert_restored(arm)


def test_gradual_returns_preserve_center_and_are_excluded_from_measured_paths(rig):
    args, arm, _ = rig
    args.experiment = 'duplicates'
    args.return_duration_s = 1
    diagnostic.run_comparison(arm, args)
    report = json.loads(next(args.output_dir.glob('*/report.json')).read_text())
    assert report['status'] == 'complete'
    assert all(t['start_positions_raw'][diagnostic.PAN] == 2000 for t in report['trials'])
    assert [t['commands_sent'] for t in report['trials']] == [101, 101, 24, 24, 101, 101]
    with next(args.output_dir.glob('*/commands.csv')).open() as f:
        rows = list(csv.DictReader(f))
    returning = [r for r in rows if r['trial'] == '2' and r['phase'] == 'return']
    assert len(returning) == 51
    assert returning[0]['intended_raw'] == '2023'
    assert returning[-1]['intended_raw'] == '2000'
    assert_restored(arm)


def test_speed_comparison_uses_per_trial_reference_time_and_restores(rig):
    args, arm, _ = rig
    args.experiment = 'speed'
    args.return_duration_s = 1
    diagnostic.run_comparison(arm, args)
    report = json.loads(next(args.output_dir.glob('*/report.json')).read_text())
    assert report['status'] == 'complete'
    assert report['durations_s'] == [2, 1, 2, 4, 2]
    assert [t['commands_sent'] for t in report['trials']] == [101, 101, 51, 51, 101, 101, 201, 201, 101, 101]
    assert all(t['pan_acceleration_raw'] == 254 for t in report['trials'])
    assert all(not t['skip_duplicates'] for t in report['trials'])
    with next(args.output_dir.glob('*/feedback.csv')).open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row['phase'] != 'test':
            continue
        progress = diagnostic.profile_progress(float(row['elapsed_s']), float(row['profile_duration_s']))
        expected = 2000 + int(row['direction'])*23*progress
        assert float(row['reference_raw']) == pytest.approx(expected)
    assert_restored(arm)


def test_dropped_target_is_logged_not_mistaken_for_tracking_error(rig):
    args, arm, _ = rig
    recorder, metadata, _ = recorder_for(arm, args)
    arm.drop = True
    commands, feedback = recorder.run(metadata, [(0., 2023., 2023)], 4)
    assert all(r['intended_raw'] == 2023 and r['goal_raw'] == r['actual_raw'] == 2000 for r in feedback)
    summary = diagnostic.summarize(commands, feedback, 2000, 2023, 1)
    assert summary['goal_mismatch_samples'] == len(feedback)
    assert summary['movement_delay_1_ticks_s'] is None
    assert summary['movement_delay_after_changed_target_s'] is None
    assert summary['motion_mean_abs_tracking_error_deg'] == pytest.approx(23*diagnostic.DEG_PER_TICK)
    assert summary['completed_travel_fraction'] == 0
    # A goal that never reaches the servo is not a pause while its goal advances.
    assert summary['central_pause_count'] == 0


@pytest.mark.parametrize('direction', [1, -1])
def test_smoothness_counts_advancing_goal_pause_jumps_and_reverse_travel(direction):
    actual = [0, 0, 0, 0, 0, 0, 3, 2, 5]
    feedback = [dict(elapsed_s=.4 + i*.04, actual_raw=2000 + direction*value,
                     goal_raw=2000 + direction*i) for i, value in enumerate(actual)]
    # Exclude endpoint observations from the motion smoothness comparison.
    feedback.extend(dict(elapsed_s=t, actual_raw=2000, goal_raw=2023) for t in (2., 2.5, 3.))
    summary = diagnostic.summarize_smoothness(feedback, 2, direction, 50)
    assert summary['central_pause_count'] == 1
    assert summary['central_pause_total_s'] == pytest.approx(.2)
    assert summary['central_longest_pause_s'] == pytest.approx(.2)
    assert summary['central_max_sample_jump_deg'] == pytest.approx(3*diagnostic.DEG_PER_TICK)
    assert summary['central_max_sample_speed_deg_s'] == pytest.approx(3*diagnostic.DEG_PER_TICK/.04)
    assert summary['central_reverse_travel_deg'] == pytest.approx(diagnostic.DEG_PER_TICK)


@pytest.mark.parametrize('advancing_goal', [False, True])
def test_short_quantized_steps_and_constant_goal_holds_are_not_pauses(advancing_goal):
    feedback = [dict(elapsed_s=.4 + i*.02, actual_raw=2000 + (i//4 if advancing_goal else 0),
                     goal_raw=2000 + (i//4 if advancing_goal else 0)) for i in range(20)]
    summary = diagnostic.summarize_smoothness(feedback, 2, 1, 50)
    assert summary['central_pause_count'] == 0


def test_sampling_gaps_split_pauses_and_do_not_create_speed_spikes():
    feedback = [dict(elapsed_s=t, actual_raw=2000 + (50 if t >= .8 else 0), goal_raw=2000+i)
                for i, t in enumerate((.4, .44, .48, .8, .84, .88))]
    summary = diagnostic.summarize_smoothness(feedback, 2, 1, 50)
    assert summary['central_sampling_gaps'] == 1
    assert summary['central_feedback_pairs'] == 4
    assert summary['central_pause_count'] == 0
    assert summary['central_max_sample_jump_deg'] == 0
    assert summary['central_max_sample_speed_deg_s'] == 0
    # Also exclude the gap if positions match on its two sides.
    for row in feedback:
        row['actual_raw'] = 2000
    assert diagnostic.summarize_smoothness(feedback, 2, 1, 50)['central_pause_count'] == 0


def test_missing_feedback_does_not_report_perfect_smoothness():
    summary = diagnostic.summarize_smoothness([], 2, 1, 50)
    assert summary['central_feedback_pairs'] == 0
    assert summary['central_pause_count'] is None
    assert summary['central_max_sample_speed_deg_s'] is None


def test_late_command_catchup_is_visible(rig):
    args, arm, clock = rig
    recorder, metadata, _ = recorder_for(arm, args)

    def slow_once():
        if len(arm.sent) == 3:
            clock.now += .08

    arm.send_hook = slow_once
    schedule = diagnostic.command_schedule(2000, 2023, 2)
    commands, feedback = recorder.run(metadata, schedule, 4)
    assert len(commands) == len(schedule)
    assert max(r['lateness_ms'] for r in commands) > 40
    assert min(r['interval_ms'] for r in commands[1:]) < 1
    assert any(r['skipped_read_slots'] > 0 for r in feedback)


def test_slow_telemetry_is_included_in_skipped_slots_and_command_lateness(rig):
    args, arm, clock = rig
    recorder, metadata, _ = recorder_for(arm, args)
    read = arm.bus.read

    def slow_voltage(register, *a, **kw):
        result = read(register, *a, **kw)
        if register == 'Present_Voltage':
            clock.now += .03
        return result

    arm.bus.read = slow_voltage
    commands, feedback = recorder.run(metadata, diagnostic.command_schedule(2000, 2023, 2), 4)
    assert len(commands) == 101
    assert all(r['skipped_read_slots'] >= 1 for r in feedback)
    assert max(r['lateness_ms'] for r in commands) > 10
    assert all(r['read_duration_ms'] > 30 for r in feedback)


@pytest.mark.parametrize('original_torque,original_lock', [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_full_mock_comparison_runs_acceleration_254_0_254_and_restores_every_setting(rig, original_torque, original_lock):
    args, arm, _ = rig
    for state in (arm.bus.registers, arm.bus.original):
        state[diagnostic.PAN].update(Torque_Enable=original_torque, Lock=original_lock)
    diagnostic.run_comparison(arm, args)
    report = json.loads(next(args.output_dir.glob('*/report.json')).read_text())
    assert report['status'] == 'complete' and report['restoration_errors'] == []
    assert report['experiment'] == 'pan_streaming_acceleration_254_0_254'
    assert report['accelerations_raw'] == [254, 0, 254]
    assert report['return_acceleration_raw'] == 254
    assert [(r['block'], r['pan_acceleration_raw'], r['direction']) for r in report['trials']] == [
        (1, 254, 1), (1, 254, -1), (2, 0, 1), (2, 0, -1), (3, 254, 1), (3, 254, -1)]
    assert {r['p_gain'] for r in report['trials']} == {48}
    assert {r['d_gain'] for r in report['trials']} == {32}
    assert {r['rate_hz'] for r in report['trials']} == {50}
    assert all(r['start_positions_raw'][diagnostic.PAN] == 2000 for r in report['trials'])
    assert len(report['block_comparison']) == 6
    assert [r['pan_acceleration_raw'] for r in report['block_comparison']] == [254, 254, 0, 0, 254, 254]
    assert all(r['median_central_pause_count'] == 0 for r in report['block_comparison'])
    assert all(r['status'] == 'verified' for r in report['settings_changes'])
    with next(args.output_dir.glob('*/commands.csv')).open() as stream:
        commands = list(csv.DictReader(stream))
    assert len(commands) == len(arm.sent_profiles)
    for row, state in zip(commands, arm.sent_profiles):
        expected = {**diagnostic.FIXED_SETTINGS, 'Acceleration': int(row['pan_acceleration_raw']), 'Lock': 1}
        assert int(row['pan_acceleration_raw']) == (0 if row['phase'] == 'test' and row['block'] == '2' else 254)
        assert int(row['p_gain']) == 48
        assert int(row['d_gain']) == 32
        for name in diagnostic.JOINTS:
            for register in diagnostic.PROTECTED:
                value = expected[register] if name == diagnostic.PAN and register in expected else arm.bus.original[name][register]
                assert state[name][register] == value
    # Identical transmitted target sequences/deadlines in all acceleration blocks.
    for direction in (1, -1):
        paths = [[(r['planned_s'], r['intended_raw']) for r in commands if r['phase'] == 'test'
                  and int(r['block']) == block and int(r['direction']) == direction] for block in (1, 2, 3)]
        assert paths[0] == paths[1] == paths[2]
    forbidden = {'Minimum_Startup_Force', 'Byte_25', 'Goal_Velocity',
                 'Maximum_Acceleration', 'Acceleration_Multiplier_Raw'}
    assert not any(reg in forbidden for reg, _, _ in arm.bus.writes)
    acceleration_writes = [v for reg, name, v in arm.bus.writes if reg == 'Acceleration']
    # Centering uses 254 between the two zero-acceleration measured trials.
    assert acceleration_writes == [254, 0, 254, 0, 254, arm.bus.original[diagnostic.PAN]['Acceleration']]
    assert_restored(arm)
    assert 'Startup_Byte_24' not in FakeBus.model_ctrl_table['sts3215']
    order = list(diagnostic.trial_order(3))
    assert len(order) == 18
    assert [r[:2] for r in order[::6]] == [(1, 254), (2, 0), (3, 254)]


def test_zero_acceleration_stall_is_measured_and_trials_still_use_the_shared_start(rig):
    args, arm, _ = rig

    def stall_at_zero():
        pan = arm.bus.registers[diagnostic.PAN]
        if pan['Acceleration'] == 0:
            pan['Present_Position'] = 2000

    arm.send_hook = stall_at_zero
    diagnostic.run_comparison(arm, args)
    report = json.loads(next(args.output_dir.glob('*/report.json')).read_text())
    assert report['status'] == 'complete'
    assert all(t['start_positions_raw'][diagnostic.PAN] == 2000 for t in report['trials'])
    for trial in report['trials']:
        if trial['pan_acceleration_raw'] == 0:
            assert trial['completed_travel_fraction'] == 0
            assert trial['movement_delay_1_ticks_s'] is None
            assert trial['central_pause_count'] == 1
            assert trial['central_longest_pause_s'] > 1
            assert trial['tail_mean_abs_endpoint_error_deg'] > 2
        else:
            assert trial['completed_travel_fraction'] == 1
    assert_restored(arm)


@pytest.mark.parametrize('failure', [ConnectionError, KeyboardInterrupt])
def test_interrupted_zero_acceleration_trial_keeps_logs_and_restores_settings(rig, failure):
    args, arm, _ = rig

    def fail():
        if arm.bus.registers[diagnostic.PAN]['Acceleration'] == 0:
            raise failure('injected send failure')

    arm.send_hook = fail
    with pytest.raises(failure):
        diagnostic.run_comparison(arm, args)
    assert_restored(arm)
    report = json.loads(next(args.output_dir.glob('*/report.json')).read_text())
    assert report['status'] == ('interrupted' if failure is KeyboardInterrupt else 'failed')
    assert report['active_trial']['block'] == 2
    with next(args.output_dir.glob('*/commands.csv')).open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[-1]['status'] == 'attempted'


def test_near_limit_rejected_before_any_motor_writes(rig):
    args, arm, _ = rig
    arm.bus.registers[diagnostic.PAN]['Present_Position'] = 3999
    with pytest.raises(RuntimeError, match='outside the calibrated test window'):
        diagnostic.run_comparison(arm, args)
    assert not arm.sent and not arm.bus.writes and not arm.bus.is_connected


def test_stalled_return_aborts_instead_of_comparing_different_starts(rig):
    args, arm, _ = rig
    recorder, metadata, _ = recorder_for(arm, args)
    arm.bus.registers[diagnostic.PAN].update(Goal_Position=2020, Present_Position=2020)
    arm.drop = True
    with pytest.raises(TimeoutError, match='return to within 1 tick'):
        recorder.run(metadata, [(0., 2000., 2000)], 1, settle=True)


@pytest.mark.parametrize('error,accepted', [(2, True), (3, False)])
def test_two_tick_start_tolerance_still_rejects_larger_errors(rig, error, accepted):
    args, arm, _ = rig
    args.start_tolerance_ticks = 2
    recorder, metadata, _ = recorder_for(arm, args)
    arm.bus.registers[diagnostic.PAN].update(Goal_Position=2000, Present_Position=2000+error)
    arm.drop = True
    if accepted:
        _, feedback = recorder.run(metadata, [(0., 2000., 2000)], 1, settle=True)
        assert feedback[-1]['actual_raw'] == 2002
    else:
        with pytest.raises(TimeoutError):
            recorder.run(metadata, [(0., 2000., 2000)], 1, settle=True)


@pytest.mark.parametrize('argv', [['--travel-deg', 'nan'], ['--duration-s', '0'],
                                  ['--read-hz', '101'], ['--repeats', '0']])
def test_invalid_arguments_rejected(argv):
    with pytest.raises(SystemExit):
        diagnostic.arguments(argv)


def test_default_is_preview_and_raw_conversion_preserves_held_positions(rig):
    args, arm, _ = rig
    assert args.execute is False
    for count in (100, 101, 2000, 3999, 4000):
        action = diagnostic.action_for_raw(dict.fromkeys(diagnostic.JOINTS, count), arm.calibration)
        encoded = arm.bus._unnormalize({m.id: action[f'{n}.pos'] for n, m in arm.bus.motors.items()})
        assert set(encoded.values()) == {count}


@pytest.mark.parametrize('register', diagnostic.TUNING_REGISTERS)
@pytest.mark.parametrize('failure', [ConnectionError, KeyboardInterrupt])
def test_partial_setting_write_restores_all_settings(rig, register, failure):
    args, arm, _ = rig
    # Ensure every requested setting differs from the original in this test.
    arm.bus.registers[diagnostic.PAN][register] = 3
    arm.bus.original[diagnostic.PAN][register] = 3
    write = arm.bus.write
    injected = False

    def fail_after_write(reg, name, value, **kwargs):
        nonlocal injected
        write(reg, name, value, **kwargs)
        if not injected and reg == register:
            injected = True
            raise failure('write reached servo but acknowledgment failed')

    arm.bus.write = fail_after_write
    with pytest.raises(failure):
        diagnostic.run_comparison(arm, args)
    assert injected
    assert_restored(arm)
    report = json.loads(next(args.output_dir.glob('*/report.json')).read_text())
    assert any(r['register'] == register and r['status'] == 'attempted' for r in report['settings_changes'])
    assert report['restoration_errors'] == []


@pytest.mark.parametrize('failure', [ConnectionError, KeyboardInterrupt])
def test_failed_zero_acceleration_write_restores_original_after_byte_was_applied(rig, failure):
    args, arm, _ = rig
    write = arm.bus.write
    injected = False

    def fail_after_zero(register, name, value, **kwargs):
        nonlocal injected
        write(register, name, value, **kwargs)
        if not injected and register == 'Acceleration' and value == 0:
            injected = True
            raise failure('zero reached servo but acknowledgment failed')

    arm.bus.write = fail_after_zero
    with pytest.raises(failure):
        diagnostic.run_comparison(arm, args)
    assert injected
    assert_restored(arm)
    report = json.loads(next(args.output_dir.glob('*/report.json')).read_text())
    assert any(e['register'] == 'Acceleration' and e['value'] == 0 and e['status'] == 'attempted'
               for e in report['settings_changes'])
    assert report['restoration_errors'] == []


def test_zero_acceleration_readback_mismatch_aborts_before_measuring_that_condition(rig):
    args, arm, _ = rig
    write = arm.bus.write

    def ignore_zero(register, name, value, **kwargs):
        write(register, name, value, **kwargs)
        if register == 'Acceleration' and value == 0:
            arm.bus.registers[name][register] = 254

    arm.bus.write = ignore_zero
    with pytest.raises(RuntimeError, match='Acceleration readback failed'):
        diagnostic.run_comparison(arm, args)
    assert_restored(arm)
    report = json.loads(next(args.output_dir.glob('*/report.json')).read_text())
    assert report['status'] == 'failed'
    assert len(report['trials']) == 2
    assert all(t['block'] == 1 for t in report['trials'])


def test_telemetry_failure_restores_settings(rig):
    args, arm, _ = rig
    read = arm.bus.read

    def fail_current(register, *a, **kw):
        if register == 'Present_Current':
            raise ConnectionError('telemetry read failed')
        return read(register, *a, **kw)

    arm.bus.read = fail_current
    with pytest.raises(ConnectionError, match='telemetry read failed'):
        diagnostic.run_comparison(arm, args)
    assert_restored(arm)


@pytest.mark.parametrize('failed_register,test_value', [('P_Coefficient', 48), ('Acceleration', 254)])
def test_cleanup_failure_is_reported_and_other_restoration_is_attempted(rig, failed_register, test_value):
    args, arm, _ = rig
    for state in (arm.bus.registers, arm.bus.original):
        state[diagnostic.PAN]['Torque_Enable'] = 1
    write = arm.bus.write
    started = False

    def fail_restore(register, name, value, **kwargs):
        nonlocal started
        if register == failed_register and name == diagnostic.PAN:
            if value == test_value:
                started = True
            elif started:
                raise ConnectionError(f'cannot restore {failed_register}')
        write(register, name, value, **kwargs)

    arm.bus.write = fail_restore

    def interrupt():
        if len(arm.sent) == 2:
            raise KeyboardInterrupt()

    arm.send_hook = interrupt
    with pytest.raises(RuntimeError, match='Cleanup could not verify'):
        diagnostic.run_comparison(arm, args)
    report = json.loads(next(args.output_dir.glob('*/report.json')).read_text())
    assert report['status'] == 'restoration_failed'
    assert any(f'restore pan {failed_register}' in error for error in report['restoration_errors'])
    for name in diagnostic.JOINTS:
        for register in (*diagnostic.PROTECTED, 'Torque_Enable'):
            if name == diagnostic.PAN and register in (failed_register, 'Torque_Enable'):
                continue
            assert arm.bus.registers[name][register] == arm.bus.original[name][register]
    assert arm.bus.registers[diagnostic.PAN]['Torque_Enable'] == 0
    assert not arm.bus.is_connected


@pytest.mark.parametrize('register,layout', [('D_Coefficient', (22, 2)),
                                          ('Acceleration', (41, 2)), ('Acceleration', (42, 1))])
def test_wrong_register_layout_aborts_without_connection(rig, register, layout):
    args, arm, _ = rig
    arm.bus.model_ctrl_table = deepcopy(FakeBus.model_ctrl_table)
    arm.bus.model_ctrl_table['sts3215'][register] = layout
    with pytest.raises(RuntimeError, match=f'{register} register layout'):
        diagnostic.run_comparison(arm, args)
    assert not arm.bus.is_connected and not arm.bus.writes and not arm.sent
