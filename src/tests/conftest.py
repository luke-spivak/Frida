"""Saved-file fixtures only; no hardware access."""

import json

import pytest


@pytest.fixture(scope="session")
def lerobot_calibration_path(tmp_path_factory):
    # Captured SO101 ranges. Tests must not depend on or modify the user's cache.
    ranges = {
        "shoulder_pan": (705, 3443), "shoulder_lift": (812, 3209),
        "elbow_flex": (916, 3114), "wrist_flex": (914, 3182),
        "wrist_roll": (0, 4095), "gripper": (2046, 3534),
    }
    path = tmp_path_factory.mktemp("lerobot-calibration") / "my_arm.json"
    path.write_text(json.dumps({
        name: {"id": i, "drive_mode": 0, "homing_offset": 1200 + i,
               "range_min": lower, "range_max": upper}
        for i, (name, (lower, upper)) in enumerate(ranges.items(), 1)
    }))
    return path
