"""Cross-region discovery and the one-session workflow; no live cloud resources."""

from dataclasses import replace
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import runpod_workflow as workflow
from remote.availability import AvailabilityError, available_placements, require_capacity
from remote.config import RunpodConfig


@pytest.fixture
def catalog(monkeypatch):
    import runpod.api.rest as rest
    def gpu(name, memory, price, region="US-WA-1", stock="HIGH"):
        return dict(id=name, memory=memory, price={"secure": price},
                    cudaVersions=[{"available": True, "version": "13.2"}],
                    dataCenters=[dict(id=region, availability=stock)])
    request = Mock(side_effect=[
        {"networkVolumes": [{"id": "volume", "dataCenter": "EUR-IS-1"}]},
        {"gpus": [gpu("NVIDIA GeForce RTX 4090", 24, .9, "EUR-IS-1", "NONE"),
                  gpu("NVIDIA GeForce RTX 4090", 24, .9),
                  gpu("NVIDIA GeForce RTX 5090", 32, 1.2, "EU-SE-1", "LOW"),
                  gpu("NVIDIA A100-SXM4-80GB", 80, 1.8, "US-KS-2"),
                  gpu("NVIDIA RTX A4000", 16, .3),
                  gpu("NVIDIA RTX A6000", 48, None),
                  gpu("NVIDIA H100 PCIe", 80, 3.0),
                  gpu("unsupported GPU", 80, .1)]}])
    monkeypatch.setattr(rest, "run_rest_request", request)
    config = RunpodConfig("example/worker:v", "volume", "EUR-IS-1")
    return config, request


def test_discovery_uses_any_region_with_storage_unchanged(catalog):
    config, request = catalog
    choices = available_placements(config)
    assert [(c["gpu"], c["datacenter"]) for c in choices] == [
        ("NVIDIA GeForce RTX 4090", "US-WA-1"),
        ("NVIDIA A100-SXM4-80GB", "US-KS-2"),
        ("NVIDIA GeForce RTX 5090", "EU-SE-1")]
    assert config.datacenter == "EUR-IS-1" and config.compute_datacenter is None
    params = request.call_args.kwargs["params"]
    assert params == {"include": "AVAILABILITY", "product": "POD", "cloud": "SECURE", "count": 1,
                      "minCudaVersion": "12.8"}
    assert all(c["cuda_versions"] == ["13.2"] for c in choices)
    assert all(call.args[0] == "GET" for call in request.call_args_list)


def test_optional_compute_pin_and_specific_gpu(catalog):
    config, _ = catalog
    choices = available_placements(replace(config, gpu="NVIDIA GeForce RTX 5090", compute_datacenter="EU-SE-1"))
    assert len(choices) == 1 and choices[0]["datacenter"] == "EU-SE-1"


@pytest.mark.parametrize("failure", ["empty", 403, 429, "timeout"])
def test_catalog_failure_does_not_launch_anything(catalog, failure):
    from runpod.error import QueryError
    config, request = catalog
    if failure == "empty":
        request.side_effect = [{"networkVolumes": [{"id": "volume", "dataCenter": "EUR-IS-1"}]}, {"gpus": []}]
    else:
        request.side_effect = TimeoutError() if failure == "timeout" else QueryError("private detail", status_code=failure)
    with pytest.raises(AvailabilityError) as exc:
        require_capacity(config)
    assert "private detail" not in str(exc.value)
    if isinstance(failure, int):
        assert f"HTTP {failure}" in str(exc.value)


def test_readonly_mode_only_prompts_api_key(catalog, monkeypatch):
    config, _ = catalog
    monkeypatch.setattr(sys, "argv", ["runpod_workflow.py", "--availability"])
    monkeypatch.setattr(RunpodConfig, "load", classmethod(lambda cls: config))
    prompt, plan = Mock(), Mock()
    monkeypatch.setattr(workflow, "prompt_credentials", prompt)
    monkeypatch.setattr(workflow, "plan_remotely", plan)
    assert workflow.main() == 0
    prompt.assert_called_once_with(api_only=True)
    plan.assert_not_called()


def test_retry_keeps_credentials_and_same_target(catalog, monkeypatch):
    config, _ = catalog
    monkeypatch.setattr(sys, "argv", ["runpod_workflow.py"])
    monkeypatch.setattr(RunpodConfig, "load", classmethod(lambda cls: config))
    prompt = Mock()
    plan = Mock(side_effect=[AvailabilityError("capacity unavailable"), Mock()])
    monkeypatch.setattr(workflow, "prompt_credentials", prompt)
    monkeypatch.setattr(workflow, "plan_remotely", plan)
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert workflow.main() == 0
    prompt.assert_called_once_with(api_only=False)
    assert plan.call_count == 2
    for call in plan.call_args_list:
        assert call.kwargs["target"] == ROOT / "targets/images.jpeg"
        assert call.kwargs["cache_dir"] == ROOT / "src/caches/so101_mvp"
        assert call.kwargs["iterations"] == 20 and call.kwargs["num_strokes"] == 400
        assert call.kwargs["config"].max_gpu_price_per_hour == 2.0


def test_ambiguous_failure_does_not_retry(catalog, monkeypatch):
    config, _ = catalog
    monkeypatch.setattr(sys, "argv", ["runpod_workflow.py"])
    monkeypatch.setattr(RunpodConfig, "load", classmethod(lambda cls: config))
    monkeypatch.setattr(workflow, "prompt_credentials", Mock())
    plan = Mock(side_effect=TimeoutError("startup timed out"))
    monkeypatch.setattr(workflow, "plan_remotely", plan)
    assert workflow.main() == 1
    plan.assert_called_once()


@pytest.mark.parametrize("arguments,expected", [
    (["--palette", "#C86E32", "#000000"], {"palette": ["#C86E32", "#000000"]}),
    (["--use_colors_from", "swatches.png", "--n_colors", "2"],
     {"use_colors_from": Path("swatches.png"), "n_colors": 2}),
])
def test_workflow_forwards_fixed_palette_options(catalog, monkeypatch, arguments, expected):
    config, _ = catalog
    monkeypatch.setattr(sys, "argv", ["runpod_workflow.py", *arguments])
    monkeypatch.setattr(RunpodConfig, "load", classmethod(lambda cls: config))
    monkeypatch.setattr(workflow, "prompt_credentials", Mock())
    plan = Mock()
    monkeypatch.setattr(workflow, "plan_remotely", plan)
    assert workflow.main() == 0
    assert expected.items() <= plan.call_args.kwargs.items()
