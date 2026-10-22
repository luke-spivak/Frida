#!/usr/bin/env python3
"""Supervised, bounded shoulder_pan velocity-mode characterization.

Preview by default; --execute enables movement. Does not use the planner.
Keeps all gains, dead zones, startup force, acceleration and factory bytes fixed.
Temporarily clears Phase bit 3 so zero speed means stop, and limits pan drive to
20% (or its original limit, if lower). Mode/Phase changes occur with pan torque off.
A zero-speed probe first uses at most 5% drive. Other joints hold measured poses.

Each pulse stops at requested travel or its deadline. A 3-degree global window,
2.5-degree per-pulse window, direction, timing, status, current, and held-joint
checks abort the run. These are software checks, not a hardware emergency stop:
host/USB failure can defeat them. A person must remain ready to cut power.
Normal stops command zero speed and observe stopping; failures disable pan torque
first. Cleanup restores configuration and torque states, with pan's position goal
set to its current measured position (never replay a stale position target).
Observed firmware 3.10 changes encoder reference and resets acceleration on mode
changes; goal writes may enable torque. The diagnostic verifies the reference
with output disabled and restores acceleration. A stationary zero-speed probe
does not guarantee a short stopping distance after motion; the stop guard remains
active, and a guard failure aborts the rest of the experiment.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time

from compare_so101_streaming import PAN, JOINTS, DEG_PER_TICK, CALIBRATION, ROOT, PROTECTED

SPEED_P = "Velocity_closed_loop_P_proportional_coefficient"
SPEED_I = "Velocity_closed_loop_I_integral_coefficient"
ALIASES = {"Startup_Byte_24": (24, 1), "Byte_25": (25, 1),
           "Acceleration_Multiplier_Raw": (86, 1), "Minimum_Speed_Raw": (83, 1)}
LAYOUT = {"Phase": (18, 1), "Operating_Mode": (33, 1), "Torque_Enable": (40, 1),
          "Homing_Offset": (31, 2), "Acceleration": (41, 1),
          "Goal_Position": (42, 2), "Goal_Velocity": (46, 2), "Torque_Limit": (48, 2),
          "Lock": (55, 1), "Present_Position": (56, 2), "Present_Velocity": (58, 2),
          "Present_Load": (60, 2), "Present_Current": (69, 2), "Present_Voltage": (62, 1),
          "Status": (65, 1), SPEED_P: (37, 1), SPEED_I: (39, 1)}
CONFIG = (*PROTECTED, SPEED_P, SPEED_I, "Min_Position_Limit", "Max_Position_Limit", "Minimum_Speed_Raw", "Homing_Offset")
SNAPSHOT = (*CONFIG, "Torque_Enable", "Present_Position", "Goal_Position",
            "Firmware_Major_Version", "Firmware_Minor_Version", "Status", "Present_Temperature")
CHANNELS = {"goal_velocity_raw": "Goal_Velocity", "velocity_raw": "Present_Velocity",
            "drive_raw": "Present_Load", "current_raw": "Present_Current",
            "voltage_raw": "Present_Voltage", "status_raw": "Status", "acceleration_raw": "Acceleration"}
FIELDS = ("trial", "phase", "command_raw", "elapsed_s", "read_start_s", "position_time_s",
          "position_register_raw", "position_raw", *(key for key in CHANNELS),
          *(key + "_time_s" for key in CHANNELS), "read_end_s", "interval_s")


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--port", default="/dev/tty.usbmodem5B8E1155331")
    p.add_argument("--calibration", type=Path, default=CALIBRATION)
    p.add_argument("--speeds-raw", nargs="+", type=int, choices=(10, 25, 50), default=[10, 25, 50])
    p.add_argument("--repeats", type=int, choices=(1, 2), default=2)
    p.add_argument("--travel-deg", type=float, default=1.5)
    p.add_argument("--max-duration-s", type=float, default=1.5)
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/so101_velocity")
    a = p.parse_args(argv)
    for key, lo, hi in (("travel_deg", .5, 1.5), ("max_duration_s", .3, 2.)):
        if not math.isfinite(getattr(a, key)) or not lo <= getattr(a, key) <= hi:
            p.error(f"{key} must be within [{lo}, {hi}]")
    return a


class SafetyStop(RuntimeError):
    pass


class Experiment:
    def __init__(self, arm, args):
        self.arm, self.bus, self.args = arm, arm.bus, args
        self.original, self.changed, self.rows = {}, False, []
        self.center = None
        self.velocity_frame = False
        self.trial = 0
        self.global_ticks = math.floor(3 / DEG_PER_TICK)
        self.travel_ticks = math.floor(args.travel_deg / DEG_PER_TICK)
        self.pulse_ticks = math.floor(2.5 / DEG_PER_TICK)
        self.output = args.output_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.output.mkdir(parents=True, exist_ok=False)
        self.report = dict(experiment="bounded_pan_velocity", status="starting", trials=[],
            arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
            events=[], restoration_errors=[], nominal_fine_speed_deg_s_per_raw=.0146*6,
            limits=dict(global_ticks=self.global_ticks, pulse_ticks=self.pulse_ticks,
                        travel_ticks=self.travel_ticks, max_active_s=args.max_duration_s,
                        maximum_drive_raw=200, zero_probe_drive_raw=50),
            note="Velocity-loop gains are unchanged. Raw speed units/precision must be validated experimentally. This is a bounded characterization, not Cartesian trajectory tracking. Software guards cannot guarantee stopping after host/USB failure.",
            source="https://ftservo.github.io/ftservo-wiki/reference/parameter/memory-sts/")
        self.epoch = time.perf_counter()

    def now(self):
        return time.perf_counter() - self.epoch

    def save(self):
        p = self.output / "report.tmp"
        p.write_text(json.dumps(self.report, indent=2) + "\n")
        p.replace(self.output / "report.json")

    def read(self, register, name=PAN):
        return int(self.bus.read(register, name, normalize=False, num_retry=0))

    def write(self, register, value, name=PAN):
        allowed = {"Torque_Enable", "Goal_Position"} if name != PAN else {
            "Torque_Enable", "Goal_Position", "Goal_Velocity", "Operating_Mode", "Phase", "Lock", "Torque_Limit",
            "Acceleration", "Homing_Offset"}
        if register not in allowed:
            raise SafetyStop(f"Write not allowed: {name} {register}")
        if register in ("Operating_Mode", "Phase") and self.read("Torque_Enable") != 0:
            raise SafetyStop("Mode/Phase changes require pan torque off")
        event = dict(time_s=self.now(), trial=self.trial, register=register, motor=name,
                     value=value, status="attempted")
        self.report['events'].append(event)
        # Durable log of attempted commands; no full-report disk I/O while moving.
        self.events.write(json.dumps(event) + "\n"); self.events.flush()
        self.bus.write(register, name, value, normalize=False, num_retry=0)
        if self.read(register, name) != value:
            raise SafetyStop(f"{name} {register} readback mismatch")
        event['status'] = 'verified'
        event['pan_torque_after'] = self.read('Torque_Enable')
        event['end_s'] = self.now()
        self.events.write(json.dumps(event) + "\n"); self.events.flush()

    def check_position(self, position, start=None, direction=0):
        cal = self.arm.calibration[PAN]
        if not cal.range_min <= position <= cal.range_max or abs(position-self.center) > self.global_ticks:
            raise SafetyStop("Global pan travel boundary exceeded")
        if start is not None and abs(position-start) > self.pulse_ticks:
            raise SafetyStop("Per-pulse travel boundary exceeded")
        if start is not None and direction and direction*(position-start) < -2:
            raise SafetyStop("Pan moved opposite to commanded direction")

    def calibrated_position(self, raw):
        # Verified on firmware 3.10 with output disabled: wheel mode reports
        # the absolute encoder before the saved homing offset; position mode
        # reports after the offset. Preserve both readings in feedback.csv.
        return (raw-self.original[PAN]['Homing_Offset']) % 4096 if self.velocity_frame else raw

    def position(self):
        return self.calibrated_position(self.read('Present_Position'))

    def capture(self, duration, phase, command=0, start=None, zero_probe=False, stop_reference=None):
        start = self.position() if start is None else start
        begin = self.now(); next_read = begin; previous = None; rows = []
        direction = (command > 0) - (command < 0)
        reason = 'time_limit'
        while True:
            if rows and self.now()-begin >= duration:
                break
            time.sleep(max(0, next_read-self.now()))
            row = dict(trial=self.trial, phase=phase, command_raw=command, read_start_s=self.now())
            row['position_register_raw'] = self.read('Present_Position')
            row['position_raw'] = self.calibrated_position(row['position_register_raw'])
            row['position_time_s'] = self.now()
            row['elapsed_s'] = row['position_time_s']-begin
            row['interval_s'] = None if previous is None else row['position_time_s']-previous
            self.report['last_position_sample'] = dict(row)
            self.check_position(row['position_raw'], start, direction)
            if row['interval_s'] is not None and row['interval_s'] > .08:
                raise SafetyStop("Position feedback gap exceeded 80 ms")
            if zero_probe and abs(row['position_raw']-start) > 1:
                raise SafetyStop("Unexpected movement at zero speed")
            if stop_reference is not None and abs(row['position_raw']-stop_reference) > 5:
                raise SafetyStop("Movement continued more than five counts after zero-speed stop")
            for key, register in CHANNELS.items():
                row[key] = self.read(register)
                row[key+'_time_s'] = self.now()
            row['read_end_s'] = self.now()
            self.writer.writerow(row); self.feedback_file.flush()
            rows.append(row); self.rows.append(row)
            if row['read_end_s']-row['read_start_s'] > .08:
                raise SafetyStop("Telemetry transaction budget exceeded 80 ms")
            if row['goal_velocity_raw'] != command:
                raise SafetyStop("Velocity command readback changed")
            if row['acceleration_raw'] != self.original[PAN]['Acceleration']:
                raise SafetyStop('Acceleration changed during velocity observation')
            if row['status_raw'] or row['current_raw']*6.5 > 400 or not 45 <= row['voltage_raw'] <= 65:
                raise SafetyStop("Servo status, current or voltage guard triggered")
            if abs(row['drive_raw']) > 210:
                raise SafetyStop("Reported drive exceeded the temporary 20% limit")
            if zero_probe and abs(row['drive_raw']) > 10:
                raise SafetyStop("Nonzero drive at zero speed")
            if len(rows) % 10 == 0:
                for name in JOINTS:
                    if name != PAN and abs(self.read('Present_Position', name)-self.original[name]['Present_Position'])*DEG_PER_TICK > 2:
                        raise SafetyStop(f"Held joint moved: {name}")
            # A rolling displacement check, less sensitive to single-count quantization.
            older = next((x for x in reversed(rows[:-1]) if row['position_time_s']-x['position_time_s'] >= .04), None)
            if older and abs(row['position_raw']-older['position_raw'])*DEG_PER_TICK/(row['position_time_s']-older['position_time_s']) > 20:
                raise SafetyStop("Measured pan speed exceeded 20 deg/s")
            if command and direction*(row['position_raw']-start) >= self.travel_ticks:
                reason = 'travel_limit'
                break
            previous = row['position_time_s']
            next_read = max(next_read+.01, self.now())
        return rows, reason

    def configure(self):
        self.bus.model_ctrl_table = {k:dict(v) for k,v in self.bus.model_ctrl_table.items()}
        table = self.bus.model_ctrl_table['sts3215']
        for key, layout in LAYOUT.items():
            if table.get(key) != layout:
                raise SafetyStop(f"Unexpected layout: {key}")
        table.update(ALIASES)
        if self.bus.model_encoding_table['sts3215'].get('Goal_Velocity') != 15:
            raise SafetyStop("Unexpected velocity sign encoding")
        self.bus.connect()  # No follower.connect/configure.
        if not self.bus.is_calibrated or set(self.arm.calibration) != set(JOINTS):
            raise SafetyStop("Calibration mismatch")
        self.original = {n:{reg:self.read(reg,n) for reg in SNAPSHOT} for n in JOINTS}
        self.report['original_registers'] = self.original
        for name, state in self.original.items():
            cal = self.arm.calibration[name]
            if state['Operating_Mode'] != 0 or state['Torque_Enable'] not in (0,1) or cal.drive_mode != 0 or cal.id != self.bus.motors[name].id:
                raise SafetyStop("Expected position-mode motors and matching non-inverted calibration")
            if not cal.range_min <= state['Present_Position'] <= cal.range_max or state['Status'] or state['Present_Temperature'] > 45:
                raise SafetyStop(f"Unsafe starting state: {name}")
        pan = self.original[PAN]
        if pan['Phase'] != 12 or pan['Angular_Resolution'] != 1 or self.bus.model_resolution_table['sts3215'] != 4096:
            raise SafetyStop("Expected previously observed Phase=12 and 4096-count pan encoder")
        self.center = pan['Present_Position']
        self.check_position(self.center-self.global_ticks); self.check_position(self.center+self.global_ticks)
        self.report['center_raw'] = self.center
        self.save()  # Original configuration saved before any write.
        self.changed = True
        self.write('Torque_Enable',0)
        self.write('Torque_Limit',0)
        for name in JOINTS:
            self.write('Goal_Position',self.read('Present_Position', name),name)
            if name != PAN and self.original[name]['Torque_Enable']==0:
                self.write('Torque_Enable',1,name)
        # Lock=1 makes EEPROM-field writes volatile on this firmware/documentation.
        self.write('Lock',1)
        # Position/lock writes can re-enable torque on the observed firmware.
        self.write('Torque_Enable',0)
        self.write('Phase',pan['Phase'] & ~8)
        self.write('Goal_Velocity',0)
        self.write('Torque_Enable',0)
        self.write('Operating_Mode',1)
        self.write('Torque_Enable',0)
        time.sleep(.06)  # Mode changes/encoder reference settle over firmware cycles.
        self.velocity_frame=True
        raw=self.read('Present_Position')
        self.report['velocity_frame_check']=dict(register_raw=raw,calibrated=self.calibrated_position(raw),homing_offset=pan['Homing_Offset'])
        if abs(self.calibrated_position(raw)-self.center)>1 or self.read('Homing_Offset')!=pan['Homing_Offset']:
            raise SafetyStop('Velocity-mode encoder reference did not match the disabled-output probe')
        # Firmware resets acceleration on a mode change; restore it explicitly.
        self.write('Acceleration',pan['Acceleration'])
        self.write('Torque_Limit',min(50,pan['Torque_Limit']))
        self.write('Torque_Enable',0)
        self.enable_at_zero()
        self.capture(.3,'zero_probe',zero_probe=True)
        self.write('Torque_Enable',0)
        self.report['zero_probe_passed'] = True
        self.write('Torque_Limit',min(200,pan['Torque_Limit']))
        self.write('Torque_Enable',0)
        self.save()

    def enable_at_zero(self):
        for key,value in {'Operating_Mode':1,'Phase':4,'Goal_Velocity':0,'Torque_Enable':0,'Lock':1}.items():
            if self.read(key) != value:
                raise SafetyStop(f"Velocity enable precondition failed: {key}")
        if not 0 < self.read('Torque_Limit') <= 200:
            raise SafetyStop('Unsafe torque limit')
        self.write('Torque_Enable',1)

    def pulse(self, command):
        self.trial += 1
        start = self.position()
        self.check_position(start)
        direction = (command>0)-(command<0)
        if abs(start + direction*(self.travel_ticks+6)-self.center) > self.global_ticks:
            raise SafetyStop('Insufficient room in global window for next pulse and stopping margin')
        self.report['active_trial'] = dict(trial=self.trial,command_raw=command,start_raw=start)
        self.save()
        self.enable_at_zero()
        self.capture(.12,'pre_pulse_zero',zero_probe=True)
        start = self.position()
        sent = self.now()
        self.write('Goal_Velocity',command)
        active,reason = self.capture(self.args.max_duration_s,'moving',command,start)
        stop_position = self.position()
        stop_time = self.now()
        self.write('Goal_Velocity',0)
        stopped,_ = self.capture(.3,'stopping',start=stop_position,stop_reference=stop_position)
        self.write('Torque_Enable',0)
        tail = [x['position_raw'] for x in stopped if x['elapsed_s'] >= .15]
        if max(tail)-min(tail)>1:
            raise SafetyStop('Velocity did not settle after zero command')
        onset = next((a['position_time_s']-sent for a,b in zip(active,active[1:])
                      if all(direction*(x['position_raw']-start)>=1 for x in (a,b))), None)
        result = dict(trial=self.trial,command_raw=command,nominal_deg_s=command*.0146*6,
            start_raw=start,end_raw=stopped[-1]['position_raw'],stop_reason=reason,
            command_time_s=sent,stop_command_time_s=stop_time,active_s=stop_time-sent,
            onset_s=onset,travel_before_stop_deg=direction*(stop_position-start)*DEG_PER_TICK,
            travel_after_stop_deg=direction*(stopped[-1]['position_raw']-stop_position)*DEG_PER_TICK,
            peak_abs_drive_pct=max(abs(x['drive_raw'])*.1 for x in active),
            peak_current_ma=max(x['current_raw']*6.5 for x in active))
        self.report['trials'].append(result); self.report.pop('active_trial',None); self.save()
        print(f"Trial {self.trial}: speed raw {command:+d}, travel {result['travel_before_stop_deg']:.3f} deg, "
              f"onset {onset}, stopped by {reason}",flush=True)

    def cleanup(self):
        def attempt(label, fn):
            try:
                fn(); return True
            except BaseException as exc:
                self.report['restoration_errors'].append(f'{label}: {type(exc).__name__}: {exc}')
                return False
        if self.changed and self.bus.is_connected:
            # On this firmware goal writes implicitly enable torque. Zero output
            # prevents motion during restoration even if a write re-enables it.
            off=attempt('disable pan torque',lambda:self.write('Torque_Enable',0))
            limited=attempt('zero output during restoration',lambda:self.write('Torque_Limit',0))
            attempt('zero pan speed',lambda:self.write('Goal_Velocity',0))
            off=attempt('disable pan before mode restoration',lambda:self.write('Torque_Enable',0)) and off
            mode_ready=(limited and off and attempt('restore Operating_Mode',lambda:self.write('Operating_Mode',0)))
            ready=mode_ready
            if mode_ready:
                time.sleep(.06)
                self.velocity_frame=False
                def restore_homing():
                    if self.read('Homing_Offset')!=self.original[PAN]['Homing_Offset']:
                        self.write('Homing_Offset',self.original[PAN]['Homing_Offset'])
                ready=attempt('verify/restore Homing_Offset',restore_homing) and ready
                def hold():
                    p=self.read('Present_Position'); cal=self.arm.calibration[PAN]
                    if not cal.range_min<=p<=cal.range_max:
                        raise SafetyStop('Cleanup position outside calibrated range')
                    self.write('Goal_Position',p)
                ready=attempt('hold current pan',hold) and ready
                ready=attempt('disable pan before Phase',lambda:self.write('Torque_Enable',0)) and ready
                for key in ('Acceleration','Phase','Goal_Velocity','Lock'):
                    ok=attempt('restore '+key,lambda key=key:self.write(key,self.original[PAN][key]))
                    ready=ready and ok
                ready=attempt('disable pan before restoring output limit',lambda:self.write('Torque_Enable',0)) and ready
                if ready:
                    ready=attempt('restore Torque_Limit',lambda:self.write('Torque_Limit',self.original[PAN]['Torque_Limit']))
            for name in JOINTS:
                torque=self.original[name]['Torque_Enable'] if name!=PAN or ready else 0
                attempt('restore torque '+name,lambda name=name,torque=torque:self.write('Torque_Enable',torque,name))
            self.report['final_registers']={}
            for name in JOINTS:
                def verify(name=name):
                    actual={reg:self.read(reg,name) for reg in (*CONFIG,'Torque_Enable')}
                    self.report['final_registers'][name]=actual
                    wrong={k:(self.original[name][k],v) for k,v in actual.items() if v!=self.original[name][k]}
                    if wrong: raise SafetyStop(str(wrong))
                attempt('verify '+name,verify)
        elif self.changed:
            self.report['restoration_errors'].append('Disconnected before cleanup; physical power cut required')
        if self.bus.is_connected:
            attempt('disconnect',lambda:self.bus.disconnect(disable_torque=False))
        if self.report['restoration_errors']:
            self.report['status']='restoration_failed'
        self.save()

    def run(self):
        with (self.output/'events.jsonl').open('w') as self.events, (self.output/'feedback.csv').open('w',newline='') as self.feedback_file:
            self.writer=csv.DictWriter(self.feedback_file,FIELDS); self.writer.writeheader()
            try:
                self.configure()
                for speed in self.args.speeds_raw:
                    for _ in range(self.args.repeats):
                        for direction in (1,-1): self.pulse(speed*direction)
                self.report['status']='complete'
            except BaseException as exc:
                self.report.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=f'{type(exc).__name__}: {exc}')
                raise
            finally:
                self.cleanup()
                print(f'Diagnostic files: {self.output}',flush=True)
                if self.report['restoration_errors']:
                    print('RESTORATION FAILED: '+str(self.report['restoration_errors']),flush=True)
        if self.report['restoration_errors']: raise SafetyStop('Restoration could not be verified')


def main(argv=None):
    args=arguments(argv)
    print(f'Pan velocity raw magnitudes {args.speeds_raw}, both directions, {args.repeats} repeats.\n'
          f'Pulse ends at {args.travel_deg} degrees or {args.max_duration_s} seconds; global software window +/-3 degrees.\n'
          'Gains unchanged. Temporary zero-means-stop Phase bit; 5% zero probe, at most 20% drive for pulses.\n'
          'All original configuration/torque states restored; pan position goal becomes measured position.')
    if not args.execute:
        print('Preview only; no robot connection.'); return
    import numpy  # Load before LeRobot/torch on macOS.
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    cal=args.calibration.expanduser().resolve(strict=True)
    arm=SO101Follower(SO101FollowerConfig(port=args.port,id=cal.stem,calibration_dir=cal.parent,
        use_degrees=True,cameras={},max_relative_target=None,disable_torque_on_disconnect=False))
    Experiment(arm,args).run()


if __name__=='__main__': main()
