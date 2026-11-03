"""Offline Runpod contract tests; no credentials, hardware, or paid resources."""

import asyncio
import gzip
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from remote import client
from remote.availability import AvailabilityError, CapacityUnavailable
from remote.artifacts import (VolumeStore, canonical, code_version, copy_cache,
                                    make_manifest, safe_path, verify_bundle, write_json)
from remote.config import RunpodConfig
from remote.settings import make_options, plan_settings
from remote.worker import _run_job, _run_transferred_job
from param2stroke import StrokeParametersToImage, get_param2img, train_param2stroke
from stroke_plan import StrokePlan


PLACEMENTS = [dict(gpu="NVIDIA GeForce RTX 4090", datacenter="US-WA-1",
                   stock="HIGH", catalog_price=.9)]


@pytest.fixture
def recorded(tmp_path):
    torch.set_num_threads(2)
    cache = tmp_path / "recorded"
    (cache / "stroke_library").mkdir(parents=True)
    materials = dict(CANVAS_WIDTH_M=.279, CANVAS_HEIGHT_M=.216,
                     CANVAS_POSITION=[0, .25], PALLETTE_POSITION=[-.2, .27], PAINT_DIFFERENCE=.04,
                     MIN_STROKE_LENGTH=.005, MAX_STROKE_LENGTH=.06, MIN_STROKE_Z=.1,
                     MAX_BEND=.02, MAX_ALPHA=0.)
    write_json(cache / "stroke_settings_during_library.json", materials)
    write_json(cache / "param2stroke_settings.json", dict(w_p2i_m=.08, h_p2i_m=.042,
               xtra_room_horz_m=.01, xtra_room_vert_m=.001, MAX_BEND=.02))
    for name in ("brush_tip_to_table_calib.json", "cached_H_canvas.pkl", "cached_H_coord.pkl"):
        (cache / name).write_bytes(b"opaque calibration bytes: never deserialized remotely")
    torch.manual_seed(3)
    torch.save(StrokeParametersToImage().state_dict(), cache / "param2img.pt")
    strokes = np.zeros((8, 216, 279), dtype=np.uint8)
    for i in range(8):
        strokes[i, 103:113, 140:155 + i * 2] = 255
    with gzip.GzipFile(cache / "stroke_library/stroke_intensities000.npy", "w") as stream:
        np.save(stream, strokes)
    parameters = np.array([[.02 + i * .003, 0, .4 + i * .05, 0] for i in range(8)], dtype=np.float32)
    np.save(cache / "stroke_library/stroke_parameters000.npy", parameters)
    material_path = tmp_path / "materials.json"
    write_json(material_path, materials)
    target = tmp_path / "target.png"
    Image.new("RGB", (64, 64), color=(140, 80, 35)).save(target)
    return SimpleNamespace(cache=cache, materials=material_path, target=target)


def mounted_job(tmp_path, recorded, kind):
    volume = tmp_path / "volume"
    snapshot = tmp_path / "snapshot"
    copy_cache(recorded.cache, snapshot / "cache")
    shutil.copyfile(recorded.materials, snapshot / "materials.json")
    write_json(snapshot / "setup.json", {"robot": "so101", "ink": False})
    manifest = make_manifest(snapshot)
    inputs = volume / "frida/inputs" / manifest["id"]
    inputs.parent.mkdir(parents=True)
    snapshot.rename(inputs)
    write_json(inputs / "manifest.json", manifest)
    job = volume / "frida/jobs/test"
    (job / "input").mkdir(parents=True)
    shutil.copyfile(recorded.target, job / "input/target.png")
    request = dict(kind=kind, planning=plan_settings(dict(objective="l2", num_strokes=4,
                   iterations=2, render_height=32, n_colors=2)),
                   training=dict(max_iterations=1, patience=1), target="target.png")
    write_json(job / "input/request.json", request)
    write_json(job / "input/manifest.json", make_manifest(job / "input"))
    packet = dict(input_prefix=f"frida/inputs/{manifest['id']}", job_prefix="frida/jobs/test",
                  source_version=code_version())
    return volume, job, packet


@pytest.mark.parametrize("kind", ["plan", "train", "train_and_plan"])
def test_real_worker_round_trip(tmp_path, recorded, kind):
    original = make_manifest(recorded.cache)
    volume, job, packet = mounted_job(tmp_path, recorded, kind)
    result = _run_job(packet, volume, "cpu")
    assert result == {"output_prefix": "frida/jobs/test/output"}
    output = job / "output"
    verify_bundle(output, json.loads((output / "manifest.json").read_text()))
    model = client.ModelResult.load(output)
    assert make_manifest(recorded.cache) == original  # Never overwrite the recorded cache.
    if kind != "plan":
        metrics = json.loads((model.cache_dir / "training_metrics.json").read_text())
        assert metrics["iterations"] == 1 and np.isfinite(metrics["best_validation_loss"])
        assert list((output / "diagnostics").rglob("*.png"))
    if kind != "train":
        opt = make_options(model.cache_dir, model.materials_json)
        opt.simulate = False  # Same setup guard used before the physical robot connects.
        plan = StrokePlan.load(output / "stroke_plan.json")
        plan.check_setup(opt)
        assert plan.remaining == 4 and plan.completed == 0
        assert Image.open(output / "preview.png").size == (41, 32)
        (model.cache_dir / "param2img.pt").write_bytes(b"changed")
        with pytest.raises(ValueError, match="changed"):
            plan.check_setup(opt)


def test_worker_rejects_stale_code_and_tampered_input(tmp_path, recorded):
    volume, _, packet = mounted_job(tmp_path, recorded, "plan")
    with pytest.raises(ValueError, match="rebuild"):
        _run_job(packet | {"source_version": "stale"}, volume, "cpu")
    (volume / packet["input_prefix"] / "cache/param2img.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        _run_job(packet, volume, "cpu")


def test_no_hardware_or_unselected_model_imports():
    # Other tests intentionally import the robot. Check the worker's imports in
    # a fresh interpreter so this contract also holds in the full test suite.
    subprocess.run([sys.executable, "-c", """
import sys
sys.path[:0] = sys.argv[1:]
import painting_optimization
assert "painter" not in sys.modules
assert "robot" not in sys.modules
assert "losses.clip_loss" not in sys.modules
""", *sys.path], check=True)


def test_renderer_allows_input_gradients_with_frozen_weights(recorded):
    opt = make_options(recorded.cache, recorded.materials)
    renderer = get_param2img(opt, "cpu")
    parameters = torch.tensor([[.03, .01, .5, 0]], requires_grad=True)
    renderer(parameters, 32, 41).sum().backward()
    assert parameters.grad is not None and torch.isfinite(parameters.grad).all()


@pytest.mark.parametrize("path", ["../secret", "/secret", "a/../../secret", "a\\secret", "a//b", ""])
def test_bundle_paths_cannot_escape(tmp_path, path):
    with pytest.raises(ValueError):
        safe_path(tmp_path, path)


def test_cache_refuses_symlinks(tmp_path, recorded):
    (recorded.cache / "param2img.pt").unlink()
    (recorded.cache / "param2img.pt").symlink_to(recorded.materials)
    with pytest.raises(ValueError, match="inside"):
        copy_cache(recorded.cache, tmp_path / "copy")


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.uploads = 0

    def get_object(self, *, Bucket, Key):
        from botocore.exceptions import ClientError
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}

    def upload_file(self, filename, bucket, key):
        self.uploads += 1
        self.objects[key] = Path(filename).read_bytes()

    def put_object(self, *, Bucket, Key, Body):
        self.objects[Key] = Body

    def download_file(self, bucket, key, filename):
        Path(filename).write_bytes(self.objects[key])


def test_transfer_deduplicates_and_verifies_before_publishing(tmp_path):
    store = VolumeStore.__new__(VolumeStore)
    store.bucket, store.client = "test", FakeS3()
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.pt").write_bytes(b"weights")
    manifest = store.upload(source, "input/version")
    assert list(store.client.objects)[-1].endswith("manifest.json")
    store.upload(source, "input/version")
    assert store.client.uploads == 1
    output = tmp_path / "download"
    store.download("input/version", output)
    verify_bundle(output, manifest)
    store.client.objects["input/version/model.pt"] = b"corrupt"
    with pytest.raises(ValueError, match="checksum"):
        store.download("input/version", tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("outcome", ["success", "error", "timeout", "cancel"])
def test_pod_lifecycle_always_cleans_up(tmp_path, outcome):
    job = SimpleNamespace(pod_id="pod-test")
    async def wait(**kwargs):
        assert kwargs["timeout"] == 10
        if outcome == "error":
            raise RuntimeError("GPU failed")
        if outcome == "timeout":
            raise TimeoutError("deadline")
        if outcome == "cancel":
            raise asyncio.CancelledError()
        return {"done": True}
    async def cancel():
        job.pod_id = None
    job.wait = wait
    job.cancel = AsyncMock(side_effect=cancel)
    task = SimpleNamespace(spawn=SimpleNamespace(aio=AsyncMock(return_value=job)))
    config = SimpleNamespace(startup_timeout=10, job_timeout=10)
    receipt = {}
    operation = client._execute(task, {}, config, receipt, tmp_path / "receipt.json")
    if outcome == "success":
        assert asyncio.run(operation) == {"done": True}
    else:
        with pytest.raises((RuntimeError, TimeoutError, asyncio.CancelledError)):
            asyncio.run(operation)
    job.cancel.assert_awaited_once()
    assert receipt["cleanup_confirmed"] is True and receipt["pod_id"] == "pod-test"


@pytest.mark.parametrize("raises", [False, True])
def test_cleanup_failure_keeps_pod_id(tmp_path, raises):
    job = SimpleNamespace(pod_id="pod-needs-cleanup", wait=AsyncMock(return_value={}),
                          cancel=AsyncMock(side_effect=OSError("offline") if raises else None))
    task = SimpleNamespace(spawn=SimpleNamespace(aio=AsyncMock(return_value=job)))
    receipt = {}
    asyncio.run(client._execute(task, {}, SimpleNamespace(startup_timeout=10, job_timeout=10),
                                receipt, tmp_path / "receipt.json"))
    assert receipt["status"] == "cleanup_required" and not receipt["cleanup_confirmed"]
    assert json.loads((tmp_path / "receipt.json").read_text())["pod_id"] == "pod-needs-cleanup"


def test_startup_failure_records_incomplete_cleanup(tmp_path):
    task = SimpleNamespace(spawn=SimpleNamespace(aio=AsyncMock(side_effect=TimeoutError())))
    receipt = {}
    with pytest.raises(TimeoutError):
        asyncio.run(client._execute(task, {}, SimpleNamespace(startup_timeout=1, job_timeout=1),
                                    receipt, tmp_path / "receipt.json"))
    assert receipt["status"] == "failed" and receipt["cleanup_confirmed"] is False


def test_placement_rejection_reports_no_pod(tmp_path, capsys):
    from runpod.apps.placement import PlacementError
    task = SimpleNamespace(spawn=SimpleNamespace(aio=AsyncMock(side_effect=PlacementError("no stock"))))
    receipt = {}
    with pytest.raises(PlacementError):
        asyncio.run(client._execute(task, {}, SimpleNamespace(startup_timeout=1, job_timeout=1),
                                    receipt, tmp_path / "receipt.json"))
    assert receipt["cleanup_confirmed"] is True and receipt["pod_created"] is False
    assert "no GPU was started" in capsys.readouterr().out


def test_real_sdk_contract_and_serialization(monkeypatch):
    import runpod
    from runpod.apps.serialization import get_function_source, serialize_arg
    from runpod.apps.tasks import _pod_input, TaskJob
    from remote.task import execute
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key-never-an-actual-credential")
    for name in ("RUNPOD_S3_ACCESS_KEY_ID", "RUNPOD_S3_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(name, "test-only-credential")
    config = RunpodConfig("example/frida:version", "volume", "US-CA-2",
                          gpu="NVIDIA A100 80GB PCIe", compute_datacenter="EUR-IS-1")
    task = client.make_task(config)
    assert hasattr(task.spawn, "aio") and asyncio.iscoroutinefunction(TaskJob.wait)
    pod = _pod_input(task.spec, "test-task-token", "test-frida")
    assert pod["gpuCount"] == 1 and pod["gpuTypeIdList"] == ["NVIDIA A100 80GB PCIe"]
    assert pod["imageName"] == config.image
    assert "runpod_sdk_runtime.task.runner" in pod["dockerArgs"]
    source = get_function_source(execute)
    namespace = {}
    exec(compile(source, "sdk-task", "exec"), namespace)
    assert callable(namespace["execute"]) and serialize_arg({"source_version": "abc"})


def test_gpu_region_is_independent_of_storage_with_real_sdk(monkeypatch):
    from runpod.apps.datacenter import DataCenter
    from runpod.apps.tasks import _pod_input, TaskExecution
    for name in ("RUNPOD_API_KEY", "RUNPOD_S3_ACCESS_KEY_ID", "RUNPOD_S3_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(name, "test-only-credential")
    config = RunpodConfig("example/frida:version", "iceland-volume", "EUR-IS-1",
                          gpu="NVIDIA GeForce RTX 4090", compute_datacenter="EU-SE-1")
    task = client.make_task(config)
    client.make_task(config)
    assert [dc.value for dc in DataCenter.all()].count("EU-SE-1") == 1
    assert not task.spec.mounts
    pod = _pod_input(task.spec, "test-token", "test-frida")
    execution = TaskExecution(task.spec)
    pod = asyncio.run(execution._attach_mounts(pod))
    assert pod["dataCenterIds"] == ["EU-SE-1"]
    assert not pod.get("volumeMounts") and not pod.get("networkVolumeId")
    env = {item["key"]: item["value"] for item in pod["env"]}
    assert env["FRIDA_CLIP_CACHE"] == "/opt/models/clip"
    assert env["RUNPOD_S3_SECRET_ACCESS_KEY"] == "test-only-credential"
    assert "FRIDA_VOLUME_ROOT" not in env


def test_live_catalog_cuda_version_reaches_deployment(monkeypatch):
    from runpod.apps import spec
    from runpod.apps.tasks import _pod_input
    from remote.availability import register_cuda_versions
    monkeypatch.setattr(spec, "CUDA_VERSIONS", frozenset({"12.8", "12.9", "13.0"}))
    register_cuda_versions(["13.2"])
    for name in ("RUNPOD_API_KEY", "RUNPOD_S3_ACCESS_KEY_ID", "RUNPOD_S3_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(name, "test-only")
    config = RunpodConfig("example/frida:v", "volume", "EUR-IS-1",
                          gpu="NVIDIA A40", compute_datacenter="EU-SE-1")
    pod = _pod_input(client.make_task(config).spec, "test-token", "test")
    assert "13.2" in pod["allowedCudaVersions"]
    with pytest.raises(ValueError):
        register_cuda_versions(["11.8"])


@pytest.mark.parametrize("settings", [{"objective": "audio"}, {"num_strokes": 123},
                                     {"iterations": 0}, {"lr_multiplier": float("nan")},
                                     {"num_strokes": 4, "n_colors": 5}])
def test_invalid_jobs_fail_locally(settings):
    with pytest.raises(ValueError):
        plan_settings(settings)


def test_fixed_palette_normalizes_colors_and_infers_count():
    settings = plan_settings({"palette": ["#C86E32", [0, 0, 0]]})
    assert settings["n_colors"] == 2
    assert settings["palette"] == [[200 / 255, 110 / 255, 50 / 255], [0., 0., 0.]]


@pytest.mark.parametrize("settings", [
    {"palette": []}, {"palette": "#123456"}, {"palette": ["orange"]},
    {"palette": [[255, 100, 0]]}, {"palette": [[0, float("nan"), 0]]},
    {"palette": [[True, 0, 0]]}, {"palette": [[0, 0]]},
    {"palette": ["#000000", [0, 0, 0]]},
    {"palette": ["#000000"], "n_colors": 2},
])
def test_bad_palettes_fail_before_cloud_access(tmp_path, recorded, monkeypatch, settings):
    capacity = Mock()
    monkeypatch.setattr(client, "require_capacity", capacity)
    with pytest.raises(ValueError):
        client.plan_remotely(recorded.target, cache_dir=recorded.cache,
                             materials_json=recorded.materials,
                             config=RunpodConfig("image:v", "volume", "EUR-IS-1"),
                             output_dir=tmp_path / "results", **settings)
    capacity.assert_not_called()


@pytest.mark.parametrize("mode", ["missing_image", "conflicting_sources", "ink"])
def test_palette_source_errors_fail_before_cloud_access(tmp_path, recorded, monkeypatch, mode):
    capacity = Mock()
    monkeypatch.setattr(client, "require_capacity", capacity)
    options = {"missing_image": {"use_colors_from": tmp_path / "missing.png"},
               "conflicting_sources": {"use_colors_from": recorded.target, "palette": ["#000000"]},
               "ink": {"ink": True, "palette": ["#000000"]}}[mode]
    with pytest.raises(ValueError):
        client.plan_remotely(recorded.target, cache_dir=recorded.cache,
                             materials_json=recorded.materials,
                             config=RunpodConfig("image:v", "volume", "EUR-IS-1"),
                             output_dir=tmp_path / "results", **options)
    capacity.assert_not_called()


@pytest.mark.parametrize("kind", ["plan", "train_and_plan"])
@pytest.mark.parametrize("palette_mode", ["explicit", "image", "image_four",
                                         "explicit_single_black", "image_single_black"])
def test_fixed_palette_survives_public_api_transfer_and_optimizer(
        tmp_path, recorded, monkeypatch, kind, palette_mode):
    expected = np.array([[200 / 255, 110 / 255, 50 / 255], [0., 0., 0.]])
    rendered_colors = []
    if palette_mode.endswith("single_black"):
        expected = np.array([[0., 0., 0.]])
        from painting import Painting
        original_forward = Painting.forward

        def check_black_before_every_render(painting, *args, **kwargs):
            colors = torch.stack([stroke.color_transform for stroke in painting.brush_strokes])
            assert torch.equal(colors, torch.zeros_like(colors))
            rendered_colors.append(colors.detach().clone())
            return original_forward(painting, *args, **kwargs)

        monkeypatch.setattr(Painting, "forward", check_black_before_every_render)
    if palette_mode == "image_four":
        expected = np.concatenate([expected, [[1., 1., 1.], [128 / 255] * 3]])
    if palette_mode.startswith("explicit"):
        options = {"palette": ["#000000"] if len(expected) == 1 else ["#C86E32", "#000000"]}
    else:
        swatches = np.zeros((256, 256, 3), dtype=np.uint8)
        for i, color in enumerate(expected):
            swatches[:, i * 256 // len(expected):(i + 1) * 256 // len(expected)] = np.rint(color * 255)
        palette_image = tmp_path / "local-swatches.png"
        Image.fromarray(swatches).save(palette_image)
        options = {"use_colors_from": palette_image}
    store = VolumeStore.__new__(VolumeStore)
    store.bucket, store.client = "test", FakeS3()
    monkeypatch.setattr(client, "VolumeStore", lambda *args, **kwargs: store)
    monkeypatch.setattr("remote.worker.VolumeStore", lambda *args, **kwargs: store)
    monkeypatch.setattr(RunpodConfig, "validate", lambda self: None)
    monkeypatch.setattr(client, "require_capacity", lambda config: PLACEMENTS)
    monkeypatch.setattr("painting.Painting.cluster_colors",
                        Mock(side_effect=AssertionError("Fixed palette must not be relearned")))
    class Job:
        pod_id = "offline-test-pod"
        def __init__(self, packet):
            self.packet = packet
        async def wait(self, timeout):
            return _run_transferred_job(self.packet, "cpu")
        async def cancel(self):
            self.pod_id = None
    async def spawn(packet):
        return Job(packet)
    monkeypatch.setattr(client, "make_task", lambda config:
                        SimpleNamespace(spawn=SimpleNamespace(aio=spawn)))
    call = client.plan_remotely if kind == "plan" else client.train_and_plan_remotely
    if kind == "train_and_plan":
        options["training"] = {"max_iterations": 1, "patience": 1}
    result = call(recorded.target, cache_dir=recorded.cache, materials_json=recorded.materials,
                  config=RunpodConfig("image:v", "volume", "EUR-IS-1"),
                  output_dir=tmp_path / "results", objective="l2", num_strokes=4,
                  iterations=2, render_height=32, n_colors=len(expected), **options)
    output = result.plan_path.parent
    palette = np.array(json.loads((output / "palette.json").read_text()))
    np.testing.assert_allclose(sorted(palette.tolist()), sorted(expected.tolist()), atol=1e-7)
    assert json.loads((output / "input_palette.json").read_text()) == palette.tolist()
    plan = StrokePlan.load(result.plan_path)
    for stroke in plan.data["strokes"]:
        np.testing.assert_allclose(stroke["color"], palette[stroke["palette_index"]], atol=1e-7)
    worker = json.loads((output / "result.json").read_text())
    if palette_mode.startswith("image"):
        assert worker["request"]["use_colors_from"] == "palette_source.png"
        assert str(palette_image) not in json.dumps(worker["request"])
    if len(expected) == 1:
        assert len(rendered_colors) >= 4  # Both loss renders, final log, exported preview.
    verify_bundle(output, json.loads((output / "manifest.json").read_text()))


def test_client_uploads_before_gpu_and_downloads_after_cleanup(tmp_path, recorded, monkeypatch):
    events = []
    monkeypatch.setattr(RunpodConfig, "validate", lambda self: None)
    class Store:
        def __init__(self, config):
            pass
        def upload(self, *args):
            events.append("upload")
        def download(self, prefix, destination):
            events.append("download")
            raise OSError("network interrupted after GPU cleanup")
    async def execute(task, packet, config, receipt, receipt_path):
        events.append("GPU cleaned up")
        return {"output_prefix": receipt["output_prefix"]}
    monkeypatch.setattr(client, "make_task", lambda config: object())
    monkeypatch.setattr(client, "require_capacity", lambda config: events.append("availability") or PLACEMENTS)
    monkeypatch.setattr(client, "VolumeStore", Store)
    monkeypatch.setattr(client, "_execute", execute)
    with pytest.raises(OSError, match="after GPU cleanup"):
        client.plan_remotely(recorded.target, cache_dir=recorded.cache, materials_json=recorded.materials,
                             config=RunpodConfig("image:v", "volume", "US-CA-2"),
                             output_dir=tmp_path / "results")
    assert events == ["availability", "upload", "upload", "GPU cleaned up", "download"]


def test_unavailable_gpu_stops_before_upload_or_spawn(tmp_path, recorded, monkeypatch):
    monkeypatch.setattr(RunpodConfig, "validate", lambda self: None)
    task = SimpleNamespace(spawn=SimpleNamespace(aio=AsyncMock()))
    store = Mock()
    monkeypatch.setattr(client, "make_task", lambda config: task)
    monkeypatch.setattr(client, "VolumeStore", store)
    monkeypatch.setattr(client, "require_capacity", Mock(side_effect=AvailabilityError("no stock")))
    with pytest.raises(AvailabilityError):
        client.plan_remotely(recorded.target, cache_dir=recorded.cache, materials_json=recorded.materials,
                             config=RunpodConfig("image:v", "volume", "EUR-IS-1"),
                             output_dir=tmp_path / "results")
    store.assert_not_called()
    task.spawn.aio.assert_not_awaited()


def test_empty_training_data_has_clear_error(recorded):
    opt = make_options(recorded.cache, recorded.materials)
    (recorded.cache / "stroke_library/stroke_parameters000.npy").unlink()
    with pytest.raises(ValueError, match="matching"):
        train_param2stroke(opt, max_iterations=1)


def test_public_training_result_can_be_used_for_next_plan(tmp_path, recorded, monkeypatch):
    volume = tmp_path / "volume"
    monkeypatch.setattr(RunpodConfig, "validate", lambda self: None)
    monkeypatch.setattr(client, "require_capacity", lambda config: PLACEMENTS)

    class MountedStore:
        def __init__(self, config=None, **kwargs):
            pass

        def upload(self, source, prefix):
            target = safe_path(volume, prefix)
            shutil.copytree(source, target, dirs_exist_ok=True)
            write_json(target / "manifest.json", make_manifest(target))

        def download(self, prefix, target):
            source = safe_path(volume, prefix)
            manifest = json.loads((source / "manifest.json").read_text())
            shutil.copytree(source, target)
            verify_bundle(target, manifest)

    class Job:
        pod_id = "offline-test-pod"

        def __init__(self, packet):
            self.packet = packet

        async def wait(self, timeout):
            return _run_transferred_job(self.packet, "cpu")

        async def cancel(self):
            self.pod_id = None

    async def spawn(packet):
        return Job(packet)

    monkeypatch.setattr(client, "VolumeStore", MountedStore)
    monkeypatch.setattr("remote.worker.VolumeStore", MountedStore)
    monkeypatch.setattr(client, "make_task", lambda config:
                        SimpleNamespace(spawn=SimpleNamespace(aio=spawn)))
    config = RunpodConfig("example/image:test", "volume", "US-CA-2")
    model = client.train_brush_remotely(recorded.cache, materials_json=recorded.materials,
                                        config=config, output_dir=tmp_path / "results",
                                        max_iterations=1)
    plan = client.plan_remotely(recorded.target, model=model, config=config,
                                output_dir=tmp_path / "results", num_strokes=4, n_colors=2,
                                iterations=1, render_height=32, objective="l2")
    assert plan.model.version == model.version
    assert plan.model.cache_dir != model.cache_dir
    receipt = json.loads(plan.receipt_path.read_text())
    assert receipt["status"] == "complete" and receipt["cleanup_confirmed"]
    assert "RUNPOD_API_KEY" not in plan.receipt_path.read_text()
    assert receipt["compute_datacenter"] == "US-WA-1"
    assert receipt["datacenter"] == "US-CA-2"
    assert receipt["input_download_seconds"] >= 0 and receipt["output_upload_seconds"] >= 0


@pytest.mark.parametrize("failure", ["capacity", "timeout", "auth", "worker"])
def test_fallback_only_after_definite_capacity_rejection(tmp_path, monkeypatch, failure):
    from runpod.error import QueryError, AuthenticationError
    choices = PLACEMENTS + [PLACEMENTS[0] | {"datacenter": "EU-SE-1"}]
    job = SimpleNamespace(pod_id="second-pod", wait=AsyncMock(return_value={"done": True}))
    async def cancel():
        job.pod_id = None
    job.cancel = AsyncMock(side_effect=cancel)
    capacity = QueryError("There are no instances currently available",
                          query="mutation { podFindAndDeployOnDemand(input: $input) { id } }")
    errors = {"capacity": capacity, "timeout": TimeoutError(), "auth": AuthenticationError("bad key")}
    first = (SimpleNamespace(spawn=SimpleNamespace(aio=AsyncMock(side_effect=errors[failure])))
             if failure != "worker" else
             SimpleNamespace(spawn=SimpleNamespace(aio=AsyncMock(return_value=job))))
    if failure == "worker":
        job.wait.side_effect = capacity  # Same text after spawn must NOT cause another deployment.
    second = SimpleNamespace(spawn=SimpleNamespace(aio=AsyncMock(return_value=job)))
    declare = Mock(side_effect=[first, second])
    monkeypatch.setattr(client, "make_task", declare)
    config = RunpodConfig("image:v", "volume", "EUR-IS-1")
    receipt = {}
    run = client._execute_candidates(choices, {}, config, receipt, tmp_path / "receipt.json")
    if failure == "capacity":
        assert asyncio.run(run) == {"done": True}
        assert declare.call_count == 2
        assert [a["status"] for a in receipt["attempts"]] == ["capacity_rejected", "completed"]
        assert receipt["cleanup_confirmed"] is True
    else:
        with pytest.raises(Exception):
            asyncio.run(run)
        assert declare.call_count == 1
        second.spawn.aio.assert_not_awaited()


def test_transfer_failure_never_reports_completed_job(tmp_path, recorded, monkeypatch):
    volume, _, packet = mounted_job(tmp_path, recorded, "plan")
    packet["storage"] = {"volume_id": "test-volume", "datacenter": "EUR-IS-1"}
    store = Mock()
    def download(prefix, target):
        shutil.copytree(volume / prefix, target)
    store.download.side_effect = download
    store.upload.side_effect = OSError("result upload failed")
    monkeypatch.setattr("remote.worker.VolumeStore", Mock(return_value=store))
    with pytest.raises(OSError, match="result upload failed"):
        _run_transferred_job(packet, "cpu")
    assert store.download.call_count == 2
    assert store.upload.call_count == 1


@pytest.mark.parametrize("kind", ["plan", "train", "train_and_plan"])
def test_s3_worker_round_trip_with_real_bundle_transfers(tmp_path, recorded, monkeypatch, kind):
    volume, _, packet = mounted_job(tmp_path, recorded, kind)
    packet["storage"] = {"volume_id": "iceland-volume", "datacenter": "EUR-IS-1"}
    store = VolumeStore.__new__(VolumeStore)
    store.bucket, store.client = "iceland-volume", FakeS3()
    store.upload(volume / packet["input_prefix"], packet["input_prefix"])
    store.upload(volume / packet["job_prefix"] / "input", f"{packet['job_prefix']}/input")
    create_store = Mock(return_value=store)
    monkeypatch.setattr("remote.worker.VolumeStore", create_store)
    result = _run_transferred_job(packet, "cpu")
    create_store.assert_called_once_with(**packet["storage"])
    # Results must survive the worker's TemporaryDirectory being deleted.
    destination = store.download(result["output_prefix"], tmp_path / "downloaded-result")
    model = client.ModelResult.load(destination)
    assert (model.cache_dir / "param2img.pt").is_file()
    if kind != "train":
        assert (destination / "preview.png").is_file()
        StrokePlan.load(destination / "stroke_plan.json").check_setup(
            make_options(model.cache_dir, model.materials_json))
