"""Portable stroke plans and atomic execution checkpoints; no hardware imports."""

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


FORMAT = "frida.stroke_plan"
VERSION = 1
DEFAULT_FILENAME = "stroke_plan.json"
SETTING_NAMES = (
    "robot", "ink", "CANVAS_WIDTH_M", "CANVAS_HEIGHT_M", "CANVAS_POSITION",
    "PALLETTE_POSITION", "PAINT_DIFFERENCE", "MIN_STROKE_LENGTH",
    "MAX_STROKE_LENGTH", "MIN_STROKE_Z", "MAX_BEND", "MAX_ALPHA",
)
CACHE_FILES = (
    "brush_tip_to_table_calib.json", "cached_H_canvas.pkl", "cached_H_coord.pkl",
    "stroke_settings_during_library.json", "param2stroke_settings.json", "param2img.pt",
)


def setup_settings(opt):
    settings = {name: copy.deepcopy(getattr(opt, name)) for name in SETTING_NAMES}
    # Painter appends calibrated Z to station coordinates during startup.
    settings["PALLETTE_POSITION"] = list(settings["PALLETTE_POSITION"][:2])
    settings["CANVAS_POSITION"] = list(settings["CANVAS_POSITION"][:2])
    return settings


def cache_fingerprints(cache_dir):
    root = Path(cache_dir)
    return {name: (hashlib.sha256((root / name).read_bytes()).hexdigest()
                   if (root / name).is_file() else None)
            for name in CACHE_FILES}


def _number(value, name, lower=-math.inf, upper=math.inf):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not lower - 1e-7 <= value <= upper + 1e-7):
        raise ValueError(f"Invalid {name}: expected a finite number in [{lower}, {upper}]")


def _rgb(color, name):
    if not isinstance(color, list) or len(color) != 3:
        raise ValueError(f"Invalid {name}: expected three RGB components")
    for component in color:
        _number(component, name, 0.0, 1.0)


class StrokePlan:
    def __init__(self, data):
        self.data = copy.deepcopy(data)
        if isinstance(self.data, dict):
            # Read plans saved before skipped-stroke tracking was introduced.
            self.data.setdefault("skipped_strokes", [])
        self.validate()

    @classmethod
    def capture(cls, painting, palette, opt, nearest_color):
        data = {
            "format": FORMAT, "version": VERSION,
            "settings": setup_settings(opt),
            "cache_fingerprints": cache_fingerprints(opt.cache_dir),
            "render_height": opt.render_height,
            "palette": None if opt.ink else palette.detach().cpu().tolist(),
            "strokes": [], "completed_strokes": 0, "in_progress": False,
            "skipped_strokes": [],
        }
        plan = cls(data)
        plan.update_remaining(painting, nearest_color)
        return plan

    @classmethod
    def load(cls, path):
        with Path(path).open() as stream:
            return cls(json.load(stream))

    @property
    def completed(self):
        return self.data["completed_strokes"]

    @property
    def remaining(self):
        return len(self.data["strokes"]) - self.completed

    @property
    def next_palette_index(self):
        return self.data["strokes"][self.completed]["palette_index"]

    def validate(self):
        d = self.data
        if (not isinstance(d, dict) or d.get("format") != FORMAT
                or type(d.get("version")) is not int or d["version"] != VERSION):
            raise ValueError("Unsupported stroke plan format/version")
        settings = d.get("settings")
        if not isinstance(settings, dict) or set(settings) != set(SETTING_NAMES):
            raise ValueError("Stroke plan is missing setup settings")
        if type(settings["ink"]) is not bool or not isinstance(settings["robot"], str):
            raise ValueError("Invalid robot/ink settings")
        for name in SETTING_NAMES[2:]:
            value = settings[name]
            if name in ("CANVAS_POSITION", "PALLETTE_POSITION"):
                if not isinstance(value, list) or len(value) != 2:
                    raise ValueError(f"Invalid {name}")
                for component in value:
                    _number(component, name)
            else:
                _number(value, name, 0.0)
        if (settings["CANVAS_WIDTH_M"] <= 0 or settings["CANVAS_HEIGHT_M"] <= 0
                or settings["MIN_STROKE_LENGTH"] > settings["MAX_STROKE_LENGTH"]
                or settings["MIN_STROKE_Z"] > 1):
            raise ValueError("Invalid canvas/stroke dimensions")
        hashes = d.get("cache_fingerprints")
        if (not isinstance(hashes, dict) or set(hashes) != set(CACHE_FILES)
                or any(value is not None and (not isinstance(value, str) or len(value) != 64)
                       for value in hashes.values())):
            raise ValueError("Invalid cache fingerprints")
        if type(d.get("render_height")) is not int or d["render_height"] < 1:
            raise ValueError("Invalid render_height")
        palette = d.get("palette")
        if settings["ink"]:
            if palette is not None:
                raise ValueError("Ink plans must not have a paint palette")
        else:
            if not isinstance(palette, list) or not palette:
                raise ValueError("Paint plans require a palette")
            for color in palette:
                _rgb(color, "palette color")
        strokes = d.get("strokes")
        if not isinstance(strokes, list):
            raise ValueError("Invalid strokes list")
        for stroke in strokes:
            if not isinstance(stroke, dict):
                raise ValueError("Invalid stroke record")
            for name in ("xt", "yt"):
                _number(stroke.get(name), name, -1.0, 1.0)
            _number(stroke.get("rotation"), "rotation")
            _number(stroke.get("length"), "length", settings["MIN_STROKE_LENGTH"],
                    settings["MAX_STROKE_LENGTH"])
            _number(stroke.get("z"), "z", settings["MIN_STROKE_Z"], 1.0)
            bend_limit = min(settings["MAX_BEND"], stroke["length"])
            _number(stroke.get("bend"), "bend", -bend_limit, bend_limit)
            _number(stroke.get("alpha"), "alpha", -settings["MAX_ALPHA"], settings["MAX_ALPHA"])
            _rgb(stroke.get("color"), "stroke color")
            index = stroke.get("palette_index")
            if settings["ink"]:
                if index is not None:
                    raise ValueError("Ink stroke must not select a palette slot")
            elif type(index) is not int or not 0 <= index < len(palette):
                raise ValueError("Stroke palette index is out of range")
            elif any(abs(a - b) > 1e-6 for a, b in zip(stroke["color"], palette[index])):
                raise ValueError("Stroke color does not match its saved palette slot")
        completed = d.get("completed_strokes")
        if type(completed) is not int or not 0 <= completed <= len(strokes):
            raise ValueError("Invalid completed_strokes")
        if type(d.get("in_progress")) is not bool or (d["in_progress"] and not self.remaining):
            raise ValueError("Invalid in_progress state")
        skipped = d.get("skipped_strokes")
        if not isinstance(skipped, list):
            raise ValueError("Invalid skipped_strokes")
        skipped_numbers = []
        for item in skipped:
            if (not isinstance(item, dict)
                    or type(item.get("stroke_number")) is not int
                    or not 1 <= item["stroke_number"] <= completed
                    or not isinstance(item.get("reason"), str)
                    or not item["reason"]):
                raise ValueError("Invalid skipped stroke record")
            skipped_numbers.append(item["stroke_number"])
        if len(set(skipped_numbers)) != len(skipped_numbers):
            raise ValueError("Duplicate skipped stroke number")

    def check_setup(self, opt):
        """Reject stale/mismatched plans before Painter can connect to hardware."""
        if not opt.use_cache:
            raise ValueError("Loading a stroke plan requires --use_cache")
        current = setup_settings(opt)
        mismatches = [name for name in SETTING_NAMES if current[name] != self.data["settings"][name]]
        if mismatches:
            raise ValueError("Stroke plan setup differs: " + ", ".join(mismatches))
        current_hashes = cache_fingerprints(opt.cache_dir)
        required = CACHE_FILES[3:] if opt.simulate else CACHE_FILES
        missing = [name for name in required if current_hashes[name] is None]
        if missing:
            raise ValueError("Cannot resume; cached files are missing: " + ", ".join(missing))
        changed = [name for name in CACHE_FILES
                   if current_hashes[name] != self.data["cache_fingerprints"][name]]
        if changed:
            raise ValueError("Stroke plan calibration/model changed: " + ", ".join(changed))
        if not (Path(opt.cache_dir) / "stroke_library").is_dir():
            raise ValueError("Saved stroke library is missing; cannot resume without collecting strokes")

    def update_remaining(self, painting, nearest_color):
        """Replace only the unpainted tail after adaptation; keep palette order."""
        import numpy as np

        palette = self.data["palette"]
        records = []
        for stroke in painting.brush_strokes:
            color = stroke.color_transform.detach().cpu().tolist()
            index = None if palette is None else int(nearest_color(
                np.asarray(color), np.asarray(palette))[0])
            records.append({
                "xt": stroke.transformation.xt.item(), "yt": stroke.transformation.yt.item(),
                "rotation": stroke.transformation.a.item(),
                "length": stroke.stroke_length.item(), "bend": stroke.stroke_bend.item(),
                "z": stroke.stroke_z.item(), "alpha": stroke.stroke_alpha.item(),
                "color": color, "palette_index": index,
            })
        self.data["strokes"] = self.data["strokes"][:self.completed] + records
        self.validate()

    def restore(self, opt, background, device):
        """Rebuild remaining strokes verbatim: no clustering, sorting or optimization."""
        import torch
        from brush_stroke import BrushStroke
        from painting import Painting

        def scalar(value):
            # BrushStroke constructs its transformation tensors on CPU; the
            # assembled Painting is moved to the requested device below.
            return torch.tensor([value], dtype=torch.float32)

        strokes = []
        for record in self.data["strokes"][self.completed:]:
            strokes.append(BrushStroke(
                opt, stroke_length=scalar(record["length"]), stroke_z=scalar(record["z"]),
                stroke_bend=scalar(record["bend"]), stroke_alpha=scalar(record["alpha"]),
                color=torch.tensor(record["color"], dtype=torch.float32, device=device),
                ink=opt.ink, a=scalar(record["rotation"]), xt=scalar(record["xt"]),
                yt=scalar(record["yt"]), device=device,
            ))
        painting = Painting(opt, background_img=background, brush_strokes=strokes).to(device)
        palette = self.data["palette"]
        palette = None if palette is None else torch.tensor(palette, dtype=torch.float32, device=device)
        return painting, palette

    def save(self, path):
        """Publish the entire checkpoint atomically; never truncate a working plan."""
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                             prefix=f".{path.name}.", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(self.data, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def begin_stroke(self, path):
        if not self.remaining:
            raise ValueError("No pending strokes")
        self.data["in_progress"] = True
        self.save(path)

    def finish_stroke(self, path):
        if not self.data["in_progress"]:
            raise ValueError("No stroke is in progress")
        self.data["completed_strokes"] += 1
        self.data["in_progress"] = False
        self.save(path)

    def skip_stroke(self, path, reason):
        """Checkpoint a planning-failed stroke as skipped and advance the plan."""
        if not self.data["in_progress"]:
            raise ValueError("No stroke is in progress")
        if not isinstance(reason, str) or not reason:
            raise ValueError("A skipped stroke requires a failure reason")
        self.data["skipped_strokes"].append({
            "stroke_number": self.completed + 1,
            "reason": reason,
        })
        self.data["completed_strokes"] += 1
        self.data["in_progress"] = False
        self.save(path)
