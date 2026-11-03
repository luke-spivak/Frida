"""Reject retired robot configurations before loading setup or opening hardware."""

import argparse
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from options import Options


def test_so101_is_default_and_simulation_remains_explicit():
    parser = Options().initialize(argparse.ArgumentParser())
    assert parser.parse_args([]).robot == 'so101'
    simulated = parser.parse_args(['--simulate'])
    assert simulated.robot == 'so101'
    assert simulated.simulate


@pytest.mark.parametrize('argv', [
    ['--robot', 'franka'], ['--robot', 'xarm'], ['--robot', 'sawyer'],
    ['--robot', 'unknown'], ['--xarm_ip', '192.168.1.176'],
])
def test_retired_robot_arguments_are_rejected(argv):
    parser = Options().initialize(argparse.ArgumentParser())
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(argv)
    assert exc.value.code == 2


def test_materials_cannot_restore_a_retired_robot(tmp_path, monkeypatch):
    materials = tmp_path / 'materials.json'
    materials.write_text(json.dumps({'robot': 'franka'}))
    monkeypatch.setattr(sys, 'argv', ['paint.py', '--materials_json', str(materials)])
    with pytest.raises(SystemExit) as exc:
        Options().gather_options()
    assert exc.value.code == 2


@pytest.mark.parametrize('robot', ['franka', 'xarm', 'sawyer', 'unknown', None])
@pytest.mark.parametrize('simulate', [False, True])
def test_painter_rejects_unsupported_setup_before_side_effects(monkeypatch, robot, simulate):
    import painter

    forbidden = Mock(side_effect=AssertionError('Unsupported setup must not open hardware'))
    for name in ('SO101', 'SimulatedRobot', 'WebCam', 'SimulatedWebCam'):
        monkeypatch.setattr(painter, name, forbidden)
    with pytest.raises(ValueError, match='Only the so101 robot is supported'):
        painter.Painter(SimpleNamespace(robot=robot, simulate=simulate))
    forbidden.assert_not_called()
