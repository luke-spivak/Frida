"""Build ordinary FRIDA options without parsing argv or touching hardware."""

import argparse
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace

from .config import ROOT

PLAN_DEFAULTS = dict(objective="clip_conv_loss", num_strokes=400, iterations=400,
                     render_height=256, n_colors=12, lr_multiplier=0.2,
                     num_augs=30, fill_weight=0.0, log_frequency=50, seed=0,
                     render_batch_size=64, cpu_threads=None, palette=None,
                     foreground_mode=False, foreground_reconstruction_weight=1.0,
                     foreground_background_weight=4.0)
TRAIN_DEFAULTS = dict(max_iterations=5000, patience=200)


def plan_settings(settings):
    unknown = set(settings) - PLAN_DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unsupported planning options: {sorted(unknown)}")
    result = PLAN_DEFAULTS | settings
    palette = result["palette"]
    if palette is not None:
        if not isinstance(palette, (list, tuple)) or not palette:
            raise ValueError("palette must contain hex colors or RGB triples in [0, 1]")
        normalized = []
        for color in palette:
            if isinstance(color, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", color):
                color = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
            if (not isinstance(color, (list, tuple)) or len(color) != 3
                    or any(type(c) not in (int, float) or not math.isfinite(c)
                           or not 0 <= c <= 1 for c in color)):
                raise ValueError("Each palette color must be #RRGGBB or three finite RGB values in [0, 1]")
            normalized.append([float(c) for c in color])
        if len({tuple(c) for c in normalized}) != len(normalized):
            raise ValueError("palette colors must be distinct")
        result["palette"] = normalized
        if "n_colors" not in settings:
            result["n_colors"] = len(normalized)
        if result["n_colors"] != len(normalized):
            raise ValueError("n_colors must match the supplied palette length")
    if result["objective"] not in ("clip_conv_loss", "clip_fc_loss", "l2", "text"):
        raise ValueError("Remote worker supports clip_conv_loss, clip_fc_loss, l2, and text")
    for name in ("num_strokes", "iterations", "render_height", "n_colors", "num_augs", "log_frequency"):
        if type(result[name]) is not int or result[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(result["render_batch_size"]) is not int or result["render_batch_size"] < 0:
        raise ValueError("render_batch_size must be a nonnegative integer")
    if result["cpu_threads"] is not None and (type(result["cpu_threads"]) is not int or result["cpu_threads"] < 1):
        raise ValueError("cpu_threads must be a positive integer or None")
    if math.isqrt(result["num_strokes"]) ** 2 != result["num_strokes"]:
        raise ValueError("FRIDA's grid initializer needs a square stroke count (e.g. 100, 225, 400)")
    if result["n_colors"] > result["num_strokes"]:
        raise ValueError("n_colors cannot exceed num_strokes")
    if type(result["foreground_mode"]) is not bool:
        raise ValueError("foreground_mode must be a boolean")
    if result["foreground_mode"] and result["objective"] == "text":
        raise ValueError("foreground_mode requires an image objective")
    for name in ("lr_multiplier", "fill_weight", "foreground_reconstruction_weight",
                 "foreground_background_weight"):
        if not isinstance(result[name], (int, float)) or not math.isfinite(result[name]) or result[name] < 0:
            raise ValueError(f"Invalid {name}")
    if type(result["seed"]) is not int or not 0 <= result["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    return result


def train_settings(settings):
    if set(settings) - TRAIN_DEFAULTS.keys():
        raise ValueError("Training options are max_iterations and patience")
    result = TRAIN_DEFAULTS | settings
    for name, value in result.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    return result


def make_options(cache_dir, materials, *, robot="so101", ink=False, planning=None):
    source = str(ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from options import Options

    defaults = vars(Options().initialize(argparse.ArgumentParser()).parse_args([]))
    material_data = json.loads(Path(materials).read_text())
    # Material files supply geometry, never paths, device choices, or job controls.
    defaults.update({k: v for k, v in material_data.items() if k.isupper()})
    defaults.update(cache_dir=str(cache_dir), materials_json=str(materials), robot=robot,
                    ink=ink, use_cache=True, simulate=True, dont_retrain_stroke_model=True)
    library = json.loads((Path(cache_dir) / "stroke_settings_during_library.json").read_text())
    for key in ("MAX_BEND", "MIN_STROKE_Z", "MIN_STROKE_LENGTH", "MAX_STROKE_LENGTH", "MAX_ALPHA"):
        defaults[key] = library[key]
    for dimension in ("WIDTH", "HEIGHT"):
        defaults[f"STROKE_LIBRARY_CANVAS_{dimension}_M"] = library[f"CANVAS_{dimension}_M"]
    if planning:
        defaults.update({k: v for k, v in planning.items() if k not in ("objective", "iterations", "seed")})
        defaults.update(objective=[planning["objective"]], objective_weight=[1.0],
                        init_optim_iter=planning["iterations"])
    opt = SimpleNamespace(**defaults)
    for name in ("CANVAS_WIDTH_M", "CANVAS_HEIGHT_M"):
        if not math.isfinite(getattr(opt, name)) or getattr(opt, name) <= 0:
            raise ValueError(f"Invalid {name}")
    from stroke_plan import SETTING_NAMES
    for name in SETTING_NAMES:
        if not hasattr(opt, name):
            raise ValueError(f"Missing material setting: {name}")
    opt.h_render = opt.render_height
    opt.w_render = int(opt.render_height * opt.CANVAS_WIDTH_M / opt.CANVAS_HEIGHT_M)
    if min(opt.h_render, opt.w_render) < 16:
        raise ValueError("Rendered canvas dimensions must be at least 16 pixels")
    return opt
