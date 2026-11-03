"""Velocity safety and restoration checks with a simulated servo; no hardware."""
from copy import deepcopy
import csv
import json
import sys
from types import SimpleNamespace

import pytest
import diagnose_so101_velocity as v


class Bus:
    is_calibrated=True
    model_resolution_table={'sts3215':4096}
    model_encoding_table={'sts3215':{'Goal_Velocity':15}}

    def __init__(self,clock):
        self.clock=clock; self.is_connected=False; self.writes=[]
        self.model_ctrl_table={'sts3215':dict(v.LAYOUT)}
        self.motors={n:SimpleNamespace(id=i,model='sts3215') for i,n in enumerate(v.JOINTS,1)}
        self.calibration={n:SimpleNamespace(id=i,range_min=100,range_max=4000,drive_mode=0) for i,n in enumerate(v.JOINTS,1)}
        self.state={n:dict.fromkeys(v.SNAPSHOT,0) for n in v.JOINTS}
        for s in self.state.values():
            s.update(Phase=12,Angular_Resolution=1,Present_Position=2000,Goal_Position=2000,
                     P_Coefficient=16,D_Coefficient=32,I_Coefficient=0,CW_Dead_Zone=1,CCW_Dead_Zone=1,
                     Startup_Byte_24=16,Acceleration=254,Torque_Limit=1000,Minimum_Speed_Raw=1,
                     Min_Position_Limit=100,Max_Position_Limit=4000,Present_Temperature=24,
                     Homing_Offset=1944,
                     Firmware_Major_Version=3,Firmware_Minor_Version=10,
                     Present_Velocity=0,Present_Load=0,Present_Current=0,Present_Voltage=55)
            s[v.SPEED_P]=10; s[v.SPEED_I]=200
        self.original=deepcopy(self.state)
        self.position=2000.; self.last_time=clock.now; self.factor=1.; self.broken_zero=False
        self.read_hook=None; self.write_hook=None

    def connect(self): self.is_connected=True
    def disconnect(self,disable_torque):
        assert disable_torque is False
        self.is_connected=False

    def advance(self):
        self.clock.now+=.0002
        s=self.state[v.PAN]; speed=0
        if s['Torque_Enable'] and s['Operating_Mode']==1 and s['Torque_Limit']>0:
            speed=s['Goal_Velocity']*self.factor
            if not speed and self.broken_zero: speed=50
        self.position+=speed*(self.clock.now-self.last_time)
        s['Present_Position']=((round(self.position)+s['Homing_Offset'])%4096
                               if s['Operating_Mode']==1 else round(self.position))
        s['Present_Velocity']=round(speed)
        s['Present_Load']=60 if speed else 0
        s['Present_Current']=2 if speed else 0
        self.last_time=self.clock.now

    def read(self,reg,name,**kwargs):
        assert self.is_connected
        assert kwargs==dict(normalize=False,num_retry=0)
        self.advance()
        if self.read_hook: self.read_hook(reg,name)
        return self.state[name][reg]

    def write(self,reg,name,value,**kwargs):
        assert self.is_connected
        assert kwargs==dict(normalize=False,num_retry=0)
        self.advance()
        if reg in ('Phase','Operating_Mode'):
            assert self.state[v.PAN]['Torque_Enable']==0
        if reg=='Torque_Enable' and name==v.PAN and value==1 and self.state[name]['Operating_Mode']==1:
            assert self.state[name]['Phase']==4
            assert self.state[name]['Goal_Velocity']==0
            assert self.state[name]['Torque_Limit']<=200
        self.writes.append((reg,name,value))
        self.state[name][reg]=value
        if reg=='Operating_Mode': self.state[name]['Acceleration']=0
        if reg=='Goal_Position' and self.state[name]['Operating_Mode']==0:
            self.state[name]['Present_Position']=value
            if name==v.PAN: self.position=float(value)
            self.state[name]['Torque_Enable']=1
        if reg=='Goal_Velocity' and self.state[name]['Operating_Mode']==1:
            self.state[name]['Torque_Enable']=1
        if self.write_hook: self.write_hook(reg,name,value)


@pytest.fixture
def rig(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'lerobot',None)
    clock=SimpleNamespace(now=0.)
    def sleep(dt): clock.now+=max(dt,1e-9)
    monkeypatch.setattr(v,'time',SimpleNamespace(perf_counter=lambda:clock.now,sleep=sleep))
    b=Bus(clock); a=SimpleNamespace(bus=b,calibration=b.calibration)
    args=v.arguments(['--output-dir',str(tmp_path),'--repeats','1','--max-duration-s','.5','--travel-deg','.75'])
    return v.Experiment(a,args),b


def assert_restored(e,b):
    assert e.report['restoration_errors']==[]
    assert not b.is_connected
    for n in v.JOINTS:
        for reg in (*v.CONFIG,'Torque_Enable'):
            assert b.state[n][reg]==b.original[n][reg]


@pytest.mark.parametrize('torque,lock',[(0,0),(1,0),(0,1),(1,1)])
def test_full_run_honors_bounds_does_not_change_gains_and_restores(rig,torque,lock):
    e,b=rig
    for state in (b.state,b.original):
        state[v.PAN].update(Torque_Enable=torque,Lock=lock,Goal_Position=3000)
    e.run()
    assert e.report['status']=='complete' and e.report['zero_probe_passed']
    assert [t['command_raw'] for t in e.report['trials']]==[10,-10,25,-25,50,-50]
    assert all(t['active_s']<=.52 and t['travel_before_stop_deg']<=.9 for t in e.report['trials'])
    assert all(abs(row['position_raw']-2000)<=e.global_ticks for row in e.rows)
    assert not any('Coefficient' in reg or 'coefficient' in reg or reg in ['Startup_Byte_24','Byte_25'] for reg,_,_ in b.writes)
    assert all(val==254 for reg,_,val in b.writes if reg=='Acceleration')
    assert any(row['position_register_raw']!=row['position_raw'] for row in e.rows)
    assert b.state[v.PAN]['Goal_Position']!=3000  # Never replay the stale goal.
    assert_restored(e,b)


def test_layout_error_never_connects_or_writes(rig):
    e,b=rig; b.model_ctrl_table['sts3215']['Goal_Velocity']=(46,1)
    with pytest.raises(v.SafetyStop,match='layout'): e.run()
    assert not b.is_connected and not b.writes


def test_bad_phase_preflight_never_writes(rig):
    e,b=rig; b.state[v.PAN]['Phase']=16
    with pytest.raises(v.SafetyStop,match='Phase=12'): e.run()
    assert not b.writes


def test_zero_command_with_unexpected_drive_aborts_before_nonzero_commands(rig):
    e,b=rig; b.broken_zero=True
    with pytest.raises(v.SafetyStop,match='zero speed'): e.run()
    assert not any(reg=='Goal_Velocity' and value!=0 for reg,_,value in b.writes)
    assert_restored(e,b)


def test_wrong_direction_disables_and_restores(rig):
    e,b=rig; b.factor=-1
    with pytest.raises(v.SafetyStop,match='opposite'): e.run()
    assert_restored(e,b)


def test_feedback_loss_disables_torque_before_restoring_mode(rig):
    e,b=rig; injected=False
    def fail(reg,name):
        nonlocal injected
        if not injected and reg=='Present_Position' and b.state[v.PAN]['Goal_Velocity']:
            injected=True; raise ConnectionError('injected feedback loss')
    b.read_hook=fail
    with pytest.raises(ConnectionError): e.run()
    last_command=max(i for i,x in enumerate(b.writes) if x[0]=='Goal_Velocity' and x[2])
    assert b.writes[last_command+1]==('Torque_Enable',v.PAN,0)
    assert_restored(e,b)


@pytest.mark.parametrize('register,value',[('Phase',4),('Operating_Mode',1),('Goal_Velocity',10),('Torque_Limit',50)])
def test_applied_write_then_interrupt_restores(rig,register,value):
    e,b=rig; injected=False
    def fail(reg,name,val):
        nonlocal injected
        if not injected and reg==register and val==value:
            injected=True; raise KeyboardInterrupt()
    b.write_hook=fail
    with pytest.raises(KeyboardInterrupt): e.run()
    assert injected
    assert_restored(e,b)


def test_phase_write_ignored_prevents_enabling_velocity_mode(rig):
    e,b=rig
    def ignore(reg,name,val):
        if reg=='Phase' and val==4: b.state[name][reg]=12
    b.write_hook=ignore
    with pytest.raises(v.SafetyStop,match='readback'): e.run()
    assert ('Operating_Mode',v.PAN,1) not in b.writes
    assert_restored(e,b)


def test_goal_writes_that_enable_torque_are_handled_before_mode_changes(rig):
    e,b=rig
    for state in (b.state,b.original): state[v.PAN]['Goal_Position']=3000
    def auto_enable(reg,name,val):
        if reg=='Goal_Position': b.state[name]['Torque_Enable']=1
    b.write_hook=auto_enable
    e.run()
    assert e.report['status']=='complete'
    assert all(abs(x['position_raw']-2000)<=e.global_ticks for x in e.rows)
    assert_restored(e,b)


def test_restore_failure_leaves_pan_unpowered_and_reports_it(rig):
    e,b=rig; b.state[v.PAN]['Torque_Enable']=b.original[v.PAN]['Torque_Enable']=1
    def fail_restore(reg,name,val):
        if reg=='Operating_Mode' and val==0:
            b.state[name][reg]=1
            raise ConnectionError('restore mode failed')
    b.write_hook=fail_restore
    with pytest.raises(v.SafetyStop,match='Restoration'): e.run()
    assert e.report['status']=='restoration_failed'
    assert b.state[v.PAN]['Torque_Enable']==0 and b.state[v.PAN]['Goal_Velocity']==0
    assert b.state[v.PAN]['Phase']==4
    assert not b.is_connected


def test_failed_speed_stop_cannot_be_reenabled_by_cleanup_position_write(rig):
    e,b=rig; started=False
    def fail_zero(reg,name,val):
        nonlocal started
        if reg=='Goal_Position': b.state[name]['Torque_Enable']=1
        if reg=='Goal_Velocity':
            if val: started=True
            elif started:
                b.state[name][reg]=10
                raise ConnectionError('zero write failed')
    b.write_hook=fail_zero
    with pytest.raises(ConnectionError): e.run()
    assert e.report['status']=='restoration_failed'
    assert b.state[v.PAN]['Torque_Enable']==0
    assert b.state[v.PAN]['Torque_Limit']==0
    assert b.state[v.PAN]['Operating_Mode']==0


def test_near_calibration_boundary_aborts_before_writes(rig):
    e,b=rig; b.position=3990
    with pytest.raises(v.SafetyStop,match='boundary'): e.run()
    assert not b.writes


def test_velocity_encoder_wrap_is_converted_before_travel_checks(rig):
    e,b=rig; b.position=2145.
    e.run()
    assert e.report['status']=='complete'
    assert any(x['position_register_raw']<20 for x in e.rows)
    assert all(abs(x['position_raw']-2145)<=e.global_ticks for x in e.rows)
    assert_restored(e,b)


def test_unexpected_velocity_frame_aborts_with_output_disabled(rig):
    e,b=rig
    def wrong_frame(reg,name):
        if reg=='Present_Position' and name==v.PAN and b.state[name]['Operating_Mode']==1:
            b.state[name]['Present_Position']=round(b.position)
    b.read_hook=wrong_frame
    with pytest.raises(v.SafetyStop,match='encoder reference'): e.run()
    assert not e.report.get('zero_probe_passed')
    assert_restored(e,b)


def test_acceleration_reset_is_restored_before_velocity_probe(rig):
    e,b=rig
    def ignore_acc(reg,name,val):
        if reg=='Acceleration' and b.state[name]['Operating_Mode']==1:
            b.state[name]['Acceleration']=0
    b.write_hook=ignore_acc
    with pytest.raises(v.SafetyStop,match='Acceleration readback'): e.run()
    assert not e.report.get('zero_probe_passed')
    assert_restored(e,b)


def test_long_feedback_gap_aborts_and_restores(rig):
    e,b=rig; injected=False
    def lag(reg,name):
        nonlocal injected
        if not injected and reg=='Present_Position' and b.state[v.PAN]['Goal_Velocity']:
            injected=True; b.clock.now+=.1
    b.read_hook=lag
    with pytest.raises(v.SafetyStop,match='80 ms'): e.run()
    assert_restored(e,b)


def test_failed_zero_stop_is_detected_and_torque_is_disabled(rig):
    e,b=rig; started=False
    def broken_stop(reg,name,value):
        nonlocal started
        if reg=='Goal_Velocity':
            if value: started=True
            elif started: b.broken_zero=True
    b.write_hook=broken_stop
    with pytest.raises(v.SafetyStop,match='after zero-speed stop'): e.run()
    assert_restored(e,b)


def test_observed_global_boundary_violation_stops_without_another_speed_command(rig):
    e,b=rig; injected=False
    def jump(reg,name):
        nonlocal injected
        if not injected and reg=='Present_Position' and b.state[v.PAN]['Goal_Velocity']:
            injected=True; b.position=2035.; b.state[v.PAN]['Present_Position']=2035
    b.read_hook=jump
    with pytest.raises(v.SafetyStop,match='Global pan'): e.run()
    assert_restored(e,b)


def test_held_joint_drift_aborts(rig):
    e,b=rig
    def drift(reg,name):
        if b.state[v.PAN]['Goal_Velocity']:
            b.state['elbow_flex']['Present_Position']=2050
    b.read_hook=drift
    with pytest.raises(v.SafetyStop,match='Held joint'): e.run()
    assert_restored(e,b)


@pytest.mark.parametrize('argv',[['--travel-deg','2'],['--max-duration-s','nan'],['--speeds-raw','100'],['--repeats','3']])
def test_invalid_args(argv):
    with pytest.raises(SystemExit): v.arguments(argv)


def test_preview_does_not_load_robot(monkeypatch,capsys):
    monkeypatch.setitem(sys.modules,'lerobot',None)
    v.main([])
    assert 'Preview only' in capsys.readouterr().out
