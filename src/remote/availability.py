"""Find GPU capacity independently of the region used for S3 storage."""

import math
import os
import re


class AvailabilityError(RuntimeError):
    pass


class CapacityUnavailable(AvailabilityError):
    """The deployment API explicitly rejected capacity before creating a Pod."""


def register_cuda_versions(versions):
    """Extend the pinned SDK with compatible versions actually advertised by Runpod."""
    from runpod.apps import spec
    for version in versions:
        if (not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+", version)
                or tuple(map(int, version.split("."))) < (12, 8)):
            raise ValueError("Invalid or incompatible catalog CUDA version")
    spec.CUDA_VERSIONS = spec.CUDA_VERSIONS | frozenset(versions)


def capacity_rejected(exc):
    from runpod.error import QueryError
    # Never retry an ambiguous timeout, authentication error, or worker failure.
    return (isinstance(exc, QueryError)
            and "podFindAndDeployOnDemand" in (exc.query or "")
            and str(exc).strip().lower().rstrip(".") in {
                "there are no instances currently available",
                "there are not enough resources to deploy your pod",
            })


def available_placements(config):
    from runpod.api.rest import run_rest_request
    from runpod.apps.gpu import GpuType

    def read(path, **kwargs):
        try:
            return run_rest_request("GET", path, api_key=os.environ.get("RUNPOD_API_KEY"), **kwargs)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            http = f", HTTP {status}" if status else ""
            raise AvailabilityError(
                f"Runpod lookup failed ({type(exc).__name__}{http}); capacity is unknown.") from None

    volumes = read("/v2/network-volumes")["networkVolumes"]
    volume = next((v for v in volumes if v["id"] == config.volume_id), None)
    if volume is None:
        raise AvailabilityError(f"Storage volume {config.volume_id} is not visible to this API key.")
    if volume["dataCenter"] != config.datacenter:
        raise AvailabilityError(
            f"Storage volume is in {volume['dataCenter']}, but runpod.json sets {config.datacenter}.")
    catalog = read("/v2/catalog/gpus", params={
        "include": "AVAILABILITY", "product": "POD", "cloud": "SECURE", "count": 1,
        "minCudaVersion": "12.8"})
    known = {gpu.value for gpu in GpuType.all()}
    preferred = ["NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 4090",
                 "NVIDIA RTX 6000 Ada Generation", "NVIDIA A100-SXM4-80GB",
                 "NVIDIA A100 80GB PCIe"]
    ranks = {gpu: rank for rank, gpu in enumerate(preferred)}
    scores = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}
    choices = []
    for gpu in catalog["gpus"]:
        price = (gpu.get("price") or {}).get("secure")
        if (gpu["id"] not in known or gpu.get("memory", 0) < 24
                or (config.gpu != "auto" and config.gpu != gpu["id"])
                or type(price) not in (float, int) or not math.isfinite(price)
                or not 0 < price <= config.max_gpu_price_per_hour):
            continue
        versions = sorted({item["version"] for item in (gpu.get("cudaVersions") or [])
                           if item.get("available") is True
                           and re.fullmatch(r"\d+\.\d+", item.get("version", ""))
                           and tuple(map(int, item["version"].split("."))) >= (12, 8)})
        for dc in gpu.get("dataCenters", []):
            if dc.get("availability") not in scores:
                raise AvailabilityError("Runpod returned an unrecognized availability status.")
            if (not scores[dc["availability"]]
                    or not re.fullmatch(r"[A-Z0-9-]+", dc["id"])
                    or (config.compute_datacenter and dc["id"] != config.compute_datacenter)):
                continue
            choices.append(dict(gpu=gpu["id"], datacenter=dc["id"], memory=gpu["memory"],
                                stock=dc["availability"], catalog_price=price, cuda_versions=versions))
    return sorted(choices, key=lambda c: (-scores[c["stock"]], ranks.get(c["gpu"], len(ranks)),
                                         c["catalog_price"], c["gpu"], c["datacenter"]))


def require_capacity(config):
    choices = available_placements(config)
    region = config.compute_datacenter or "all regions"
    print(f"S3 storage: {config.volume_id} in {config.datacenter}; GPU search: {region}.")
    if not choices:
        raise AvailabilityError(
            f"No matching GPUs reported under the ${config.max_gpu_price_per_hour:g}/hour "
            "catalog limit (24+ GB, CUDA 12.8+, Secure Cloud). No GPU started.")
    print(f"Found {len(choices)} GPU/region options under the "
          f"${config.max_gpu_price_per_hour:g}/hour catalog limit (rates are estimates).")
    return choices
