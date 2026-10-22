"""Record one relative lift and one straight stroke; default is OFFLINE.

    python diagnose_so101_motion.py
    python diagnose_so101_motion.py --execute
    python diagnose_so101_motion.py --execute --read-every 0

Execution starts at the current brush position: +20 mm Z, then +40 mm FRIDA X.
Position the brush in clear air before running with --execute. There is no
approach, descent, or return move. The normal planner and executor are reused.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

SOURCE_DIR = Path(__file__).resolve().parent


class MotionLog:
    """Instrument existing calls; restore all methods even if a move fails."""

    def __init__(self, robot, read_every=1, offline_observation=None):
        self.robot = robot
        self.read_every = read_every
        self.offline_observation = offline_observation
        self.epoch = time.perf_counter()
        self.planned, self.commands, self.feedback, self.events = [], [], [], []

    def move(self, phase, positions):
        robot = self.robot
        planner = robot.motion_planner
        plan, execute = planner.plan_xyz_path, robot.execute_trajectory
        send, read = robot.arm.send_action, robot.arm.get_observation
        trajectory = None
        active = None
        event = dict(phase=phase, target_frida_m=np.asarray(positions).tolist(),
                     planning_start_s=time.perf_counter() - self.epoch)
        self.events.append(event)

        def capture_plan(*args, **kwargs):
            nonlocal trajectory
            trajectory = plan(*args, **kwargs)
            event['planning_end_s'] = time.perf_counter() - self.epoch
            return trajectory

        def capture_read(kind="settling"):
            begin = time.perf_counter()
            observation = read()
            end = time.perf_counter()
            if active is None:
                event['start_observation'] = {name: float(value) for name, value in observation.items() if name.endswith('.pos')}
            if active is not None and active['index'] >= 0:
                row = dict(phase=phase, kind=kind, sample=active['index'],
                           read_start_s=begin-self.epoch, read_end_s=end-self.epoch,
                           time_s=(begin+end)/2-self.epoch, read_ms=1000*(end-begin))
                for name in robot.arm_joint_names:
                    row[f'{name}_actual_deg'] = float(observation[f'{name}.pos'])
                    row[f'{name}_error_deg'] = float(active['target'][f'{name}.pos']) - row[f'{name}_actual_deg']
                self.feedback.append(row)
            return observation

        def capture_send(action):
            active['index'] += 1
            index = active['index']
            begin = time.perf_counter()
            if active['origin'] is None:
                # Anchor the schedule to the first actual send. This measures
                # accumulated lateness, not latency before the first command.
                active['origin'] = begin - active['times'][index]
            result = send(action)
            end = time.perf_counter()
            row = dict(phase=phase, sample=index, planned_s=float(active['times'][index]),
                       send_s=begin-self.epoch, write_ms=1000*(end-begin),
                       lateness_ms=1000*(begin-active['origin']-active['times'][index]),
                       interval_ms=1000*(begin-active['previous']) if index else '',
                       expected_interval_ms=1000*(active['times'][index]-active['times'][index-1]) if index else '')
            row.update({f'{n}_target_deg': float(action[f'{n}.pos']) for n in robot.arm_joint_names})
            self.commands.append(row)
            active['previous'], active['target'] = begin, action.copy()
            if self.read_every and index % self.read_every == 0:
                capture_read("tracking")
            return result

        def capture_execute(times, actions, *args, **kwargs):
            nonlocal active
            event['execution_start_s'] = time.perf_counter() - self.epoch
            event['duration_s'] = float(times[-1])
            for index, (t, action, velocity, acceleration) in enumerate(zip(
                    times, actions, trajectory.velocities, trajectory.accelerations)):
                row = dict(phase=phase, sample=index, time_s=float(t))
                for j, name in enumerate(trajectory.joint_names):
                    row[f'{name}_target_deg'] = float(action[f'{name}.pos'])
                    row[f'{name}_velocity_deg_s'] = float(np.rad2deg(velocity[j]))
                    row[f'{name}_acceleration_deg_s2'] = float(np.rad2deg(acceleration[j]))
                self.planned.append(row)
            print(f"{phase.upper()}: planned {times[-1]:.3f}s; "
                  f"{'offline only' if self.offline_observation is not None else 'starting motion'}", flush=True)
            if self.offline_observation is not None:
                # Ideal endpoint propagation for planning only; never fabricate
                # measured timings or tracking-error rows in offline mode.
                self.offline_observation.update(actions[-1])
            else:
                active = dict(index=-1, times=times, origin=None, previous=None, target=None)
                execute(times, actions, *args, **kwargs)
            event['execution_end_s'] = time.perf_counter() - self.epoch

        planner.plan_xyz_path = capture_plan
        robot.execute_trajectory = capture_execute
        robot.arm.send_action = capture_send
        robot.arm.get_observation = capture_read
        try:
            robot.go_to_cartesian_pose(positions)
            event['status'] = 'complete' if trajectory is not None else 'no_motion'
        except BaseException as exc:
            event['status'], event['error'] = 'failed', f'{type(exc).__name__}: {exc}'
            raise
        finally:
            event['end_s'] = time.perf_counter() - self.epoch
            planner.plan_xyz_path, robot.execute_trajectory = plan, execute
            robot.arm.send_action, robot.arm.get_observation = send, read

    def summary(self):
        report = {}
        for phase in ('lift', 'stroke'):
            commands = [r for r in self.commands if r['phase'] == phase]
            feedback = [r for r in self.feedback if r['phase'] == phase and r['kind'] == 'tracking']
            info = dict(commands_sent=len(commands), tracking_reads=len(feedback))
            if len(commands) > 1:
                intervals = [r['interval_ms'] for r in commands[1:]]
                info.update(interval_ms_min=float(min(intervals)), interval_ms_p95=float(np.percentile(intervals, 95)),
                            interval_ms_max=float(max(intervals)), max_lateness_ms=max(r['lateness_ms'] for r in commands),
                            write_ms_p95=float(np.percentile([r['write_ms'] for r in commands], 95)))
            if feedback:
                info['read_ms_p95'] = float(np.percentile([r['read_ms'] for r in feedback], 95))
                info['tracking_error_deg'] = {
                    name: dict(rms=float(np.sqrt(np.mean([r[f'{name}_error_deg']**2 for r in feedback]))),
                               max_abs=max(abs(r[f'{name}_error_deg']) for r in feedback))
                    for name in self.robot.arm_joint_names
                }
            report[phase] = info
        return report

    def save(self, directory, metadata):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, rows in [('planned', self.planned), ('commands', self.commands), ('feedback', self.feedback)]:
            if rows:
                with (directory / f'{name}.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
        report = dict(**metadata, events=self.events, measurements=self.summary(),
                      timing_note='Lateness is relative to the first send of each move. Read timestamps bracket a bus transaction, not simultaneous encoder sampling.',
                      tracking_note='Error = most recently sent target minus measured position, in servo degrees. Measurements add bus load; read-every=0 is the timing control.')
        (directory / 'report.json').write_text(json.dumps(report, indent=2))
        self.plot(directory / 'diagnostic.png')
        print(f"Diagnostic files: {directory.resolve()}", flush=True)

    def plot(self, path):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 2, figsize=(12, 12), squeeze=False)
        names = self.robot.arm_joint_names
        for column, phase in enumerate(('lift', 'stroke')):
            planned = [r for r in self.planned if r['phase'] == phase]
            commands = [r for r in self.commands if r['phase'] == phase]
            feedback = [r for r in self.feedback if r['phase'] == phase]
            if planned:
                times = [r['time_s'] for r in planned]
                for name in names:
                    axes[0, column].plot(times, [r[f'{name}_velocity_deg_s'] for r in planned], label=name)
                    axes[1, column].plot(times, [r[f'{name}_acceleration_deg_s2'] for r in planned])
            if commands:
                origin = commands[0]['send_s']
                axes[2, column].plot([r['send_s']-origin for r in commands[1:]],
                                     [r['interval_ms'] for r in commands[1:]], label='actual interval')
                axes[2, column].plot([r['send_s']-origin for r in commands[1:]],
                                     [r['expected_interval_ms'] for r in commands[1:]], '--', label='planned interval')
                axes[2, column].plot([r['time_s']-origin for r in feedback],
                                     [r['read_ms'] for r in feedback], '.', label='feedback read time')
                for name in names:
                    axes[3, column].plot([r['time_s']-origin for r in feedback],
                                         [r[f'{name}_error_deg'] for r in feedback])
                axes[2, column].legend(fontsize=7)
            else:
                for ax in axes[2:, column]:
                    ax.text(.5, .5, 'Offline: no measured data', ha='center', transform=ax.transAxes)
            axes[0, column].set_title(phase.upper())
            for row, label in enumerate(('Planned velocity (°/s)', 'Planned acceleration (°/s²)',
                                         'Command/read timing (ms)', 'Last goal − measured (°)')):
                axes[row, column].set(xlabel='Seconds within move', ylabel=label)
                axes[row, column].grid(alpha=.25)
        if self.planned:
            axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)


def run_diagnostic(robot, log, lift_mm=20.0, stroke_mm=40.0, axis='x'):
    start = np.asarray(robot.get_cartesian_pose(), dtype=float)
    lifted = start + [0, 0, lift_mm / 1000]
    log.move('lift', [lifted])
    # Re-read the achieved lift position: the stroke should start where the arm
    # actually is, without a hidden corrective Z move in this diagnostic.
    stroke_start = np.asarray(robot.get_cartesian_pose(), dtype=float)
    end = stroke_start.copy()
    end['xy'.index(axis)] += stroke_mm / 1000
    points = np.linspace(stroke_start, end, max(2, int(np.ceil(abs(stroke_mm) / 3)) + 1))
    log.move('stroke', points)


def offline_robot(start_file):
    from robot import SO101
    from motion_planner import SO101MotionPlanner
    saved = json.loads(Path(start_file).read_text())
    observation = saved.get('observation', saved).copy()
    robot = SO101.__new__(SO101)
    robot.motion_planner = SO101MotionPlanner()
    robot.arm_joint_names = robot.motion_planner.joint_names
    def no_hardware(*args, **kwargs):
        raise AssertionError('Hardware execution is forbidden in offline mode')
    robot.arm = SimpleNamespace(get_observation=lambda: observation.copy(), send_action=no_hardware)
    return robot, observation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--execute', action='store_true', help='Connect and perform the two relative moves')
    parser.add_argument('--read-every', type=int, default=1, help='Read after every N sends; 0 logs send timing only (settling reads remain)')
    parser.add_argument('--lift-mm', type=float, default=20)
    parser.add_argument('--stroke-mm', type=float, default=40, help='Signed displacement along the selected FRIDA axis')
    parser.add_argument('--axis', choices=['x', 'y'], default='x')
    parser.add_argument('--start-joints', type=Path, default=SOURCE_DIR / 'so101_offsets.json', help='Saved observation for OFFLINE replay only')
    parser.add_argument('--output', type=Path, help='New output directory (default: outputs/so101_diagnostic/UTC timestamp)')
    args = parser.parse_args(argv)
    if args.read_every < 0 or not np.isfinite(args.lift_mm) or args.lift_mm <= 0 or not np.isfinite(args.stroke_mm) or args.stroke_mm == 0:
        parser.error('Use read-every >= 0, a positive finite lift, and a nonzero finite stroke')
    stamp = datetime.now(timezone.utc)
    output = (args.output or SOURCE_DIR.parent / 'outputs/so101_diagnostic' / stamp.strftime('%Y%m%dT%H%M%S%fZ')).resolve()
    args.start_joints = args.start_joints.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(SOURCE_DIR)
    if args.execute:
        from robot import SO101
        robot, observation = SO101(debug=True, show_viewer=False), None
    else:
        robot, observation = offline_robot(args.start_joints)
    log = MotionLog(robot, args.read_every, observation)
    metadata = dict(mode='hardware' if args.execute else 'offline', started_utc=stamp.isoformat(),
                    read_every=args.read_every, lift_mm=args.lift_mm, stroke_mm=args.stroke_mm,
                    axis=args.axis, start_joints_file=str(args.start_joints) if not args.execute else None,
                    joint_offsets_deg=robot.motion_planner.joint_offsets_deg,
                    calibration_file=str(robot.motion_planner.lerobot_calibration_path))
    try:
        if args.execute:
            robot.good_morning_robot()
        else:
            print('OFFLINE: planning from saved joints. No motor connection or timing/tracking measurements.')
        run_diagnostic(robot, log, args.lift_mm, args.stroke_mm, args.axis)
    finally:
        try:
            log.save(output, metadata)
        finally:
            if args.execute and robot.arm.bus.is_connected:
                robot.good_night_robot()


if __name__ == '__main__':
    main()
