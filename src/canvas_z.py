"""Canvas contact-height interpolation and cache I/O; no hardware imports."""

import json
import os
from pathlib import Path
import tempfile

import numpy as np


HEIGHT_FILENAME = "brush_tip_to_table_calib.json"
GRID_KEY = "canvas_z_grid"
DEFAULT_INSET_M = 0.02


def canvas_geometry(opt):
    position = np.asarray(opt.CANVAS_POSITION[:2], dtype=float)
    width, height = float(opt.CANVAS_WIDTH_M), float(opt.CANVAS_HEIGHT_M)
    if (position.shape != (2,) or not np.all(np.isfinite(position))
            or not np.isfinite([width, height]).all() or min(width, height) <= 0):
        raise ValueError("Invalid canvas position/dimensions for Z calibration")
    return position, width, height


def grid_axes(opt, inset_m=DEFAULT_INSET_M):
    """Three command-space coordinates per axis, inside the canvas edges."""
    position, width, height = canvas_geometry(opt)
    if not np.isfinite(inset_m) or not 0 <= inset_m < min(width, height) / 2:
        raise ValueError("Z calibration inset must be nonnegative and less than half the canvas size")
    return (
        np.linspace(position[0] - width / 2 + inset_m,
                    position[0] + width / 2 - inset_m, 3),
        np.linspace(position[1] + inset_m, position[1] + height - inset_m, 3),
    )


class CanvasZCalibration:
    """Bilinear absolute contact heights, indexed [y row, x column].

    Coordinates are the FRIDA XYZ requests sent to the robot, after any camera
    XY correction. Outside the sampled rectangle, hold the nearest edge value;
    never extrapolate the slope into the stations or beyond measured samples.
    """

    def __init__(self, opt, contact_z_m, inset_m=DEFAULT_INSET_M):
        self.x_m, self.y_m = grid_axes(opt, inset_m)
        self.contact_z_m = np.array(contact_z_m, dtype=float, copy=True)
        if self.contact_z_m.shape != (3, 3) or not np.isfinite(self.contact_z_m).all():
            raise ValueError("Canvas Z calibration requires nine finite contact heights (3 x 3)")
        position, width, height = canvas_geometry(opt)
        self.geometry = {
            "robot": opt.robot, "canvas_position_m": position.tolist(),
            "canvas_width_m": width, "canvas_height_m": height,
            "inset_m": float(inset_m),
        }

    def to_dict(self):
        return {
            "version": 1, "frame": "frida_command_xy", **self.geometry,
            "contact_z_m": self.contact_z_m.tolist(),
        }

    @classmethod
    def from_dict(cls, data, opt):
        try:
            if (data["version"] != 1 or data["frame"] != "frida_command_xy"
                    or data["robot"] != opt.robot):
                raise ValueError("Unsupported canvas Z calibration version/frame/robot")
            position, width, height = canvas_geometry(opt)
            saved = np.asarray([*data["canvas_position_m"], data["canvas_width_m"],
                                data["canvas_height_m"]], dtype=float)
            if (saved.shape != (4,) or not np.isfinite(saved).all()
                    or not np.allclose(saved, [*position, width, height], rtol=0, atol=1e-9)):
                raise ValueError("Canvas settings changed; recalibrate the canvas Z grid")
            return cls(opt, data["contact_z_m"], float(data["inset_m"]))
        except (KeyError, TypeError) as exc:
            raise ValueError("Invalid canvas Z calibration data") from exc

    def contact_z(self, x, y):
        if not np.isfinite([x, y]).all():
            raise ValueError("Canvas Z interpolation requires finite XY coordinates")
        x = float(np.clip(x, self.x_m[0], self.x_m[-1]))
        y = float(np.clip(y, self.y_m[0], self.y_m[-1]))
        col = min(int(np.searchsorted(self.x_m, x, side="right")) - 1, 1)
        row = min(int(np.searchsorted(self.y_m, y, side="right")) - 1, 1)
        u = (x - self.x_m[col]) / (self.x_m[col + 1] - self.x_m[col])
        v = (y - self.y_m[row]) / (self.y_m[row + 1] - self.y_m[row])
        z = self.contact_z_m
        return float((1-v) * ((1-u)*z[row, col] + u*z[row, col+1])
                     + v * ((1-u)*z[row+1, col] + u*z[row+1, col+1]))


def read_height_calibration(path, opt, load_grid=True):
    with Path(path).open() as stream:
        params = json.load(stream)
    try:
        heights = [float(params[name]) for name in ("Z_CANVAS", "Z_MAX_CANVAS")]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Height calibration needs Z_CANVAS and Z_MAX_CANVAS") from exc
    if not np.isfinite(heights).all():
        raise ValueError("Height calibration must contain finite heights")
    grid = (CanvasZCalibration.from_dict(params[GRID_KEY], opt)
            if load_grid and GRID_KEY in params else None)
    return params, grid


def save_grid(path, params, grid):
    """Replace the cache atomically only after a complete grid was collected.

    Retain the scalar reference/pressure range and station heights. The existing
    stroke-plan fingerprint of this file also binds plans to this grid.
    """
    path = Path(path)
    updated = {**params, GRID_KEY: grid.to_dict()}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = stream.name
            json.dump(updated, stream, indent=4, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
