"""Local orchestration: upload first, run one task, terminate, download results."""

import asyncio
from dataclasses import dataclass, replace
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import uuid

from .artifacts import VolumeStore, code_version, copy_cache, make_manifest, write_json
from .availability import require_capacity, CapacityUnavailable, capacity_rejected, register_cuda_versions
from .config import ROOT, SDK_VERSION, RunpodConfig
from .settings import make_options, plan_settings, train_settings


@dataclass(frozen=True)
class ModelResult:
    version: str
    cache_dir: Path
    materials_json: Path
    robot: str
    ink: bool

    @classmethod
    def load(cls, directory):
        directory = Path(directory).resolve()
        data = json.loads((directory / "result.json").read_text())
        return cls(data["model_version"], directory / "cache", directory / "materials.json",
                   data["robot"], data["ink"])


@dataclass(frozen=True)
class PlanResult:
    plan_path: Path
    preview_path: Path
    model: ModelResult
    receipt_path: Path


def make_task(config):
    import runpod
    from .task import execute

    if importlib.metadata.version("runpod") != SDK_VERSION:
        raise RuntimeError(f"Install src/remote/requirements-client.txt (requires runpod=={SDK_VERSION})")
    if config.gpu == "auto" or not config.compute_datacenter:
        raise ValueError("Select a GPU and compute datacenter from the catalog before declaring a task")
    # The pinned SDK's region enum is incomplete. Register the exact region
    # selected from the live catalog, including regions without network storage.
    from runpod.apps.datacenter import DataCenter
    dc = config.compute_datacenter
    if dc not in DataCenter._value2member_map_:
        region = str.__new__(DataCenter, dc)
        region._name_, region._value_ = dc.replace("-", "_"), dc
        setattr(DataCenter, region._name_, region)
        DataCenter._member_map_[region._name_] = region
        DataCenter._value2member_map_[region._value_] = region
        DataCenter._member_names_.append(region._name_)
    app = runpod.App("frida")
    task = app.task(
        name="frida_gpu_job", gpu=config.gpu, gpu_count=1, image=config.image,
        registry_auth=config.registry_auth, datacenter=dc,
        container_disk_gb=config.container_disk_gb,
        min_cuda_version="12.8", accelerate_downloads=False,
        env={"MPLBACKEND": "Agg",
             # The runtime watchdog needs this to delete its own Pod if the
             # client disappears. Never written to image, artifacts, or receipts.
             "RUNPOD_API_KEY": os.environ["RUNPOD_API_KEY"],
             "RUNPOD_S3_ACCESS_KEY_ID": os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
             "RUNPOD_S3_SECRET_ACCESS_KEY": os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"],
             "RUNPOD_TASK_IDLE_TIMEOUT": "120",
             "FRIDA_CLIP_CACHE": "/opt/models/clip",
             "TORCH_HOME": "/tmp/frida-pretrained/torch",
             "HF_HOME": "/tmp/frida-pretrained/huggingface"},
    )(execute)
    task.spec.validate()
    return task


async def _execute(task, packet, config, receipt, receipt_path):
    from runpod.apps.placement import PlacementError

    job = None
    placement_rejected = False
    started = time.monotonic()
    try:
        receipt["status"] = "starting"
        write_json(receipt_path, receipt)
        # The SDK cleans up provisioning failures/cancellation before spawn returns.
        job = await asyncio.wait_for(task.spawn.aio(packet), timeout=config.startup_timeout)
        receipt.update(pod_id=job.pod_id, status="running",
                       startup_seconds=time.monotonic() - started)
        write_json(receipt_path, receipt)
        result = await job.wait(timeout=config.job_timeout)
        receipt["status"] = "computed"
        return result
    except BaseException as exc:
        # In the pinned SDK, volume placement runs before deploy_task_pod.
        rejected_capacity = job is None and capacity_rejected(exc)
        placement_rejected = job is None and (isinstance(exc, PlacementError) or rejected_capacity)
        receipt.update(status="interrupted" if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt))
                       else "failed", error_type=type(exc).__name__)
        if rejected_capacity:
            raise CapacityUnavailable("Runpod rejected this GPU/region before Pod creation.") from None
        raise
    finally:
        if job is not None:
            try:
                if job.pod_id:
                    await job.cancel()  # Retry if SDK wait's best-effort cleanup failed.
            except Exception as exc:
                receipt["cleanup_error_type"] = type(exc).__name__
            finally:
                receipt["cleanup_confirmed"] = job.pod_id is None
                if job.pod_id:
                    receipt.update(status="cleanup_required", pod_id=job.pod_id)
                    print(f"Pod cleanup is unconfirmed. Terminate {job.pod_id} in the Runpod console.")
        elif placement_rejected:
            receipt.update(cleanup_confirmed=True, pod_created=False)
            print("Placement failed before Pod creation; no GPU was started.")
        else:
            receipt["cleanup_confirmed"] = False
            print("Startup did not finish. The SDK attempts cleanup; check Runpod for task-frida_gpu_job Pods.")
        receipt["task_wall_seconds"] = time.monotonic() - started
        write_json(receipt_path, receipt)


async def _execute_candidates(choices, packet, config, receipt, receipt_path):
    """Retry only definite capacity rejections; never duplicate an ambiguous job."""
    receipt["attempts"] = []
    for choice in choices[:config.max_placement_attempts]:
        register_cuda_versions(choice.get("cuda_versions", []))
        selected = replace(config, gpu=choice["gpu"], compute_datacenter=choice["datacenter"])
        task = make_task(selected)
        receipt.pop("pod_created", None)
        receipt.pop("error_type", None)
        receipt.update(gpu=selected.gpu, compute_datacenter=selected.compute_datacenter)
        attempt = {**choice, "status": "starting"}
        receipt["attempts"].append(attempt)
        print(f"Trying {selected.gpu} in {selected.compute_datacenter} "
              f"(~${choice['catalog_price']:.2f}/hour; {choice['stock']} stock).")
        try:
            result = await _execute(task, packet, selected, receipt, receipt_path)
        except CapacityUnavailable:
            attempt["status"] = "capacity_rejected"
            print("Trying the next available GPU/region using the same uploaded inputs.")
        except BaseException:
            attempt["status"] = "failed"
            raise
        else:
            attempt["status"] = "completed"
            return result
        finally:
            write_json(receipt_path, receipt)
    raise CapacityUnavailable("All candidate deployments were rejected for capacity; no GPU was started.")


def _run(kind, *, target=None, model=None, cache_dir=None, materials_json=None,
         robot="so101", ink=False, config=None, output_dir=None, planning=None, training=None,
         use_colors_from=None):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("These helpers are synchronous; in a notebook use await asyncio.to_thread(helper, ...)")
    config = config or RunpodConfig.load()
    planning = plan_settings(planning or {}) if kind != "train" else None
    training = train_settings(training or {}) if kind != "plan" else None
    if model is not None:
        if not isinstance(model, ModelResult):
            raise TypeError("model must be a ModelResult (or ModelResult.load(result_directory))")
        if cache_dir is not None or materials_json is not None:
            raise ValueError("Use either model or cache_dir/materials_json")
        cache_dir, materials_json, robot, ink = model.cache_dir, model.materials_json, model.robot, model.ink
    cache_dir = Path(cache_dir or ROOT / "src/caches/so101_mvp").resolve()
    materials_json = Path(materials_json or ROOT / "my_materials.json").resolve()
    if not isinstance(robot, str) or not robot or type(ink) is not bool:
        raise ValueError("robot must be a name and ink must be a bool")
    if use_colors_from is not None:
        if planning is None or planning["palette"] is not None:
            raise ValueError("use_colors_from requires planning and cannot be combined with palette")
        use_colors_from = Path(use_colors_from).resolve()
        if not use_colors_from.is_file():
            raise ValueError(f"Palette image does not exist: {use_colors_from}")
    if ink and planning and (planning["palette"] is not None or use_colors_from is not None):
        raise ValueError("A paint palette cannot be used with ink=True")
    output_dir = Path(output_dir or ROOT / "outputs/runpod").resolve()
    job_id = uuid.uuid4().hex
    destination = output_dir / job_id
    receipt_path = output_dir / f"{job_id}.json"
    with tempfile.TemporaryDirectory(prefix="frida-upload-") as directory:
        staging = Path(directory)
        inputs, job_input = staging / "model", staging / "job"
        inputs.mkdir()
        job_input.mkdir()
        copy_cache(cache_dir, inputs / "cache", training=kind != "plan")
        shutil.copyfile(materials_json, inputs / "materials.json")
        write_json(inputs / "setup.json", {"robot": robot, "ink": ink})
        # Check material geometry and option validity before allocating a paid GPU.
        make_options(inputs / "cache", inputs / "materials.json", robot=robot, ink=ink, planning=planning)
        request = {"kind": kind, "planning": planning, "training": training}
        if planning:
            if use_colors_from is not None:
                palette_name = "palette_source" + use_colors_from.suffix.lower()
                shutil.copyfile(use_colors_from, job_input / palette_name)
                request["use_colors_from"] = palette_name
            if target is None or not Path(target).is_file():
                raise ValueError(f"Target image does not exist: {target}")
            target_name = "target" + Path(target).suffix.lower()
            shutil.copyfile(target, job_input / target_name)
            request["target"] = target_name
        write_json(job_input / "request.json", request)
        config.validate()
        source_version = code_version()
        image_tag = config.image.rsplit(":", 1)[-1]
        if re.fullmatch(r"[0-9a-f]{16}", image_tag) and image_tag != source_version[:16]:
            raise ValueError("Configured image is stale; rebuild it before starting a GPU")
        print("Checking storage and GPU availability across regions (no GPU running yet).")
        choices = require_capacity(config)
        store = VolumeStore(config)
        version = make_manifest(inputs)["id"]
        if model and version != model.version:
            raise ValueError("This model's cache or materials changed; use its original version")
        input_prefix = f"{config.prefix}/inputs/{version}"
        job_prefix = f"{config.prefix}/jobs/{job_id}"
        packet = {"input_prefix": input_prefix, "job_prefix": job_prefix,
                  "source_version": source_version,
                  "storage": {"volume_id": config.volume_id, "datacenter": config.datacenter}}
        receipt = {"job_id": job_id, "status": "uploading", "image": config.image,
                   "gpu": config.gpu, "volume_id": config.volume_id, "datacenter": config.datacenter,
                   "output_prefix": f"{job_prefix}/output", "destination": str(destination),
                   "source_version": packet["source_version"], "input_version": version}
        write_json(receipt_path, receipt)
        print(f"Uploading inputs for {job_id} (no GPU running yet).")
        store.upload(inputs, input_prefix)
        store.upload(job_input, f"{job_prefix}/input")
        print(f"Starting one GPU task; receipt: {receipt_path}")
        result = asyncio.run(_execute_candidates(choices, packet, config, receipt, receipt_path))
        for name in ("input_download_seconds", "output_upload_seconds"):
            if name in result:
                receipt[name] = result[name]
        if result.get("output_prefix") != receipt["output_prefix"]:
            raise ValueError("Worker returned an unexpected output location")
        # Download happens after Pod termination. Network-volume files survive it.
        store.download(receipt["output_prefix"], destination)
    model_result = ModelResult.load(destination)
    if kind != "train":
        from stroke_plan import StrokePlan
        opt = make_options(model_result.cache_dir, model_result.materials_json,
                           robot=robot, ink=ink, planning=planning)
        StrokePlan.load(destination / "stroke_plan.json").check_setup(opt)
    receipt["status"] = "complete" if receipt.get("cleanup_confirmed") else "cleanup_required"
    write_json(receipt_path, receipt)
    print(f"Results saved to {destination}")
    if kind == "train":
        return model_result
    return PlanResult(destination / "stroke_plan.json", destination / "preview.png", model_result, receipt_path)


def plan_remotely(target, *, model=None, cache_dir=None, materials_json=None,
                  robot="so101", ink=False, config=None, output_dir=None, use_colors_from=None,
                  **planning):
    """Plan remotely; optionally use palette=[hex/RGB colors] or a palette image."""
    return _run("plan", target=target, model=model, cache_dir=cache_dir, materials_json=materials_json,
                robot=robot, ink=ink, config=config, output_dir=output_dir, planning=planning,
                use_colors_from=use_colors_from)


def train_brush_remotely(cache_dir=None, *, materials_json=None, robot="so101", ink=False,
                         config=None, output_dir=None, max_iterations=5000, patience=200):
    """Train recorded strokes, returning a separate cache with the new model."""
    return _run("train", cache_dir=cache_dir, materials_json=materials_json, robot=robot, ink=ink,
                config=config, output_dir=output_dir,
                training={"max_iterations": max_iterations, "patience": patience})


def train_and_plan_remotely(target, *, cache_dir=None, materials_json=None, robot="so101", ink=False,
                            config=None, output_dir=None, training=None, use_colors_from=None, **planning):
    """Train, then plan using the new model in the same Pod (one startup)."""
    return _run("train_and_plan", target=target, cache_dir=cache_dir, materials_json=materials_json,
                robot=robot, ink=ink, config=config, output_dir=output_dir,
                training=training, planning=planning, use_colors_from=use_colors_from)


def recover_results(receipt_path, *, config=None):
    """Retry an interrupted download without launching another GPU."""
    config = config or RunpodConfig.load()
    config.validate()
    receipt = json.loads(Path(receipt_path).read_text())
    if receipt["volume_id"] != config.volume_id or receipt["datacenter"] != config.datacenter:
        raise ValueError("Receipt belongs to a different volume/datacenter")
    return VolumeStore(config).download(receipt["output_prefix"], receipt["destination"])
