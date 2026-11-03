"""Exercise the interactive read-only utility without opening a serial port."""

from unittest.mock import Mock, call

import pytest

import position_util


@pytest.fixture
def reader(monkeypatch):
    robot = Mock()
    robot.arm.config.port = "unused-port"
    original_motors = robot.arm.bus.motors
    constructor = Mock(return_value=robot)
    bus = Mock(is_connected=True, is_calibrated=True)
    bus_factory = Mock(return_value=bus)
    monkeypatch.setattr(position_util, "SO101", constructor)
    monkeypatch.setattr(position_util, "ReadOnlyFeetechBus", bus_factory)
    return robot, constructor, bus, bus_factory, original_motors


def test_prints_fresh_frida_positions_using_getter(reader, monkeypatch, capsys):
    robot, constructor, bus, bus_factory, motors = reader
    robot.get_cartesian_pose.side_effect = [[-0.2, 0.1, 0.08], [-0.21, 0.12, 0.09]]
    monkeypatch.setattr("builtins.input", Mock(side_effect=["water", "rag", " Q "]))

    position_util.main()

    output = capsys.readouterr().out
    assert "'label': 'water', 'frida_xyz_m': [-0.2, 0.1, 0.08]" in output
    assert "'label': 'rag', 'frida_xyz_m': [-0.21, 0.12, 0.09]" in output
    assert robot.get_cartesian_pose.call_count == 2
    constructor.assert_called_once_with(debug=True, show_viewer=False)
    bus_factory.assert_called_once_with("unused-port", motors, calibration=robot.arm.calibration)
    assert robot.arm.bus is bus
    assert bus.method_calls == [call.connect(), call.disconnect(disable_torque=False)]
    robot.good_morning_robot.assert_not_called()
    robot.good_night_robot.assert_not_called()
    robot.arm.connect.assert_not_called()
    robot.arm.configure.assert_not_called()
    robot.arm.send_action.assert_not_called()
    robot.arm.disconnect.assert_not_called()


@pytest.mark.parametrize("outcome", [
    "calibration_mismatch", "handshake_error", "port_error", "read_error", "interrupt", "eof",
])
def test_exit_and_failures_preserve_torque(reader, monkeypatch, outcome):
    robot, _, bus, _, _ = reader
    inputs = Mock(side_effect=["point"])
    monkeypatch.setattr("builtins.input", inputs)
    if outcome == "calibration_mismatch":
        bus.is_calibrated = False
    elif outcome in ("handshake_error", "port_error"):
        bus.connect.side_effect = ConnectionError("connection failed")
        bus.is_connected = outcome == "handshake_error"
    elif outcome == "read_error":
        robot.get_cartesian_pose.side_effect = ConnectionError("read failed")
    else:
        inputs.side_effect = KeyboardInterrupt if outcome == "interrupt" else EOFError

    if outcome in ("interrupt", "eof"):
        position_util.main()
    else:
        with pytest.raises((RuntimeError, ConnectionError)):
            position_util.main()

    expected = [call.connect()]
    if bus.is_connected:
        expected.append(call.disconnect(disable_torque=False))
    assert bus.method_calls == expected
    if outcome != "read_error":
        robot.get_cartesian_pose.assert_not_called()
    robot.good_morning_robot.assert_not_called()
    robot.good_night_robot.assert_not_called()
    robot.arm.connect.assert_not_called()
    robot.arm.send_action.assert_not_called()
