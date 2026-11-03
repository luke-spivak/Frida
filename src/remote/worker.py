"""CUDA worker. Never imports Painter, robot drivers, or camera drivers."""

import json
import os
from pathlib import Path
import shutil
import time
import tempfile

from .artifacts import (copy_cache, make_manifest, safe_path, verify_bundle,
                        code_version, write_json, VolumeStore)
from .settings import make_options, plan_settings, train_settings


class ArtifactWriter:
    """Keep diagnostics as ordinary PNG/JSON files for easy local review."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def add_image(self, tag, image, step=0):
        import numpy as np
        from PIL import Image
        image = np.asarray(image)
        if image.max() <= 1:
            image = image * 255
        path = safe_path(self.directory, f"{tag}-{step}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.clip(image, 0, 255).astype("uint8")).save(path)

    def add_scalar(self, tag, value, step=0):
        with (self.directory / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps({"tag": tag, "step": step, "value": float(value)}, allow_nan=False) + "\n")

    def add_figure(self, tag, figure, step=0):
        from matplotlib import pyplot as plt
        path = safe_path(self.directory, f"{tag}-{step}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path)
        plt.close(figure)


def _bundle(root):
    manifest = json.loads((root / "manifest.json").read_text())
    verify_bundle(root, manifest)
    return manifest


def run_job(packet):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This worker requires CUDA; refusing a silent CPU fallback")
    return _run_transferred_job(packet, "cuda")


def _run_transferred_job(packet, device):
    """Stage S3 bundles on local Pod disk; publish results before Pod deletion."""
    if packet["source_version"] != code_version():
        raise ValueError("Worker image does not match local FRIDA code; rebuild and publish it")
    store = VolumeStore(**packet["storage"])
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="frida-job-") as directory:
        root = Path(directory)
        store.download(packet["input_prefix"], safe_path(root, packet["input_prefix"]))
        job = safe_path(root, packet["job_prefix"])
        store.download(f"{packet['job_prefix']}/input", job / "input")
        download_seconds = time.monotonic() - started
        try:
            result = _run_job(packet, root, device)
        except Exception:
            if (job / "working").exists():
                try:
                    store.upload(job / "working", f"{packet['job_prefix']}/failed")
                except Exception as exc:
                    print(f"Could not save failure diagnostics: {type(exc).__name__}")
            raise
        before_upload = time.monotonic()
        store.upload(job / "output", result["output_prefix"])
        return {**result, "input_download_seconds": download_seconds,
                "output_upload_seconds": time.monotonic() - before_upload}


def _run_job(packet, volume_root, device):
    # device is injectable only here for small offline tests. The SDK entry point
    # above always requires CUDA.
    started = time.monotonic()
    if packet["source_version"] != code_version():
        raise ValueError("Worker image does not match local FRIDA code; rebuild and publish it")
    inputs = safe_path(volume_root, packet["input_prefix"])
    job = safe_path(volume_root, packet["job_prefix"])
    input_manifest = _bundle(inputs)
    if inputs.name != input_manifest["id"]:
        raise ValueError("Input version does not match its manifest")
    _bundle(job / "input")
    request = json.loads((job / "input/request.json").read_text())
    if request["kind"] not in ("plan", "train", "train_and_plan"):
        raise ValueError("Unknown job kind")
    setup = json.loads((inputs / "setup.json").read_text())
    output = job / "output"
    if output.exists():
        _bundle(output)
        return {"output_prefix": f"{packet['job_prefix']}/output"}
    staging = job / "working"
    staging.mkdir(exist_ok=False)
    copy_cache(inputs / "cache", staging / "cache", training=request["kind"] != "plan")
    shutil.copyfile(inputs / "materials.json", staging / "materials.json")
    planning = plan_settings(request["planning"]) if request["kind"] != "train" else None
    opt = make_options(staging / "cache", staging / "materials.json", **setup, planning=planning)
    opt.writer = ArtifactWriter(staging / "diagnostics")
    timing = {}
    if request["kind"] != "plan":
        from param2stroke import train_param2stroke
        before = time.monotonic()
        train_param2stroke(opt, device=device, **train_settings(request["training"]))
        timing["training_seconds"] = time.monotonic() - before
    # Model identity excludes diagnostics; it includes calibration and materials.
    # Same structure as the local input snapshot, so subsequent jobs reuse it.
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        snapshot = Path(directory)
        copy_cache(staging / "cache", snapshot / "cache")
        shutil.copyfile(staging / "materials.json", snapshot / "materials.json")
        write_json(snapshot / "setup.json", setup)
        model_version = make_manifest(snapshot)["id"]
    if planning:
        before = time.monotonic()
        _plan(opt, planning, request["target"], job / "input", staging, device,
              use_colors_from=request.get("use_colors_from"))
        timing["planning_seconds"] = time.monotonic() - before
    import torch
    result = {"model_version": model_version, "input_version": input_manifest["id"],
              "source_version": packet["source_version"], "request": request, **setup,
              "device": str(device), "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
              "torch_version": torch.__version__, **timing,
              "worker_seconds": time.monotonic() - started}
    write_json(staging / "result.json", result)
    write_json(staging / "manifest.json", make_manifest(staging))
    staging.rename(output)
    return {"output_prefix": f"{packet['job_prefix']}/output"}


def _plan(opt, planning, target, inputs, output, device, *, use_colors_from=None):
    import numpy as np
    import random
    import torch
    from PIL import Image
    from paint_utils3 import nearest_color, random_init_painting
    import painting_optimization as optimization
    from stroke_plan import StrokePlan

    if planning.get("cpu_threads") is not None:
        torch.set_num_threads(planning["cpu_threads"])

    random.seed(planning["seed"])
    np.random.seed(planning["seed"])
    torch.manual_seed(planning["seed"])
    palette = None
    if use_colors_from is not None:
        if planning.get("palette") is not None:
            raise ValueError("Use either palette or use_colors_from")
        import cv2
        from paint_utils3 import get_colors
        image_path = safe_path(inputs, use_colors_from)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("Cannot decode the palette image")
        # Match paint.py's existing image-to-palette path exactly.
        palette = get_colors(cv2.resize(image[:, :, ::-1], (256, 256)),
                             n_colors=planning["n_colors"]).to(device)
    elif planning.get("palette") is not None:
        palette = torch.tensor(planning["palette"], dtype=torch.float32, device=device)
    if palette is not None:
        if opt.ink:
            raise ValueError("A paint palette cannot be used with ink=True")
        write_json(output / "input_palette.json", palette.detach().cpu().tolist())
    # CLIP weights are baked into the image; no mounted volume is needed.
    cache = os.environ.get("FRIDA_CLIP_CACHE")
    if cache:
        Path(cache).mkdir(parents=True, exist_ok=True)
        for weight in Path("/opt/models/clip").glob("*.pt"):
            destination = Path(cache) / weight.name
            if not destination.exists():
                temporary = destination.with_name(destination.name + "." + output.parent.name + ".tmp")
                shutil.copyfile(weight, temporary)
                temporary.replace(destination)
    opt.objective_data = [target if planning["objective"] == "text" else str(safe_path(inputs, target))]
    optimization.local_it = 0
    optimization.plans = []
    optimization.load_objectives_data(opt)
    background = torch.ones(1, 3, opt.h_render, opt.w_render, device=device)
    painting = random_init_painting(opt, background, planning["num_strokes"], ink=opt.ink, device=device)
    painting, palette = optimization.optimize_painting(opt, painting, planning["iterations"],
                                                      color_palette=palette)
    if hasattr(painting, "foreground_diagnostics"):
        write_json(output / "foreground_diagnostics.json", painting.foreground_diagnostics)
    plan = StrokePlan.capture(painting, palette, opt, nearest_color)
    plan.check_setup(opt)
    plan.save(output / "stroke_plan.json")
    with torch.no_grad():
        preview = painting(opt.h_render, opt.w_render, use_alpha=False)[0, :3]
        pixels = preview.detach().cpu().permute(1, 2, 0).numpy()
    Image.fromarray((np.clip(pixels, 0, 1) * 255).astype("uint8")).save(output / "preview.png")
    if palette is not None:
        write_json(output / "palette.json", palette.detach().cpu().tolist())
