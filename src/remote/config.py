"""Non-secret deployment settings; credentials stay in the local environment."""

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
REMOTE_DIR = Path(__file__).resolve().parent
SDK_VERSION = "2.0.0rc1"


@dataclass(frozen=True)
class RunpodConfig:
    image: str
    volume_id: str
    datacenter: str
    gpu: str = "auto"
    registry_auth: str | None = None
    prefix: str = "frida"
    startup_timeout: int = 900
    job_timeout: int = 1800
    container_disk_gb: int = 30
    compute_datacenter: str | None = None
    max_gpu_price_per_hour: float = 2.0
    max_placement_attempts: int = 12

    @classmethod
    def load(cls, path=None):
        path = Path(path or os.environ.get("FRIDA_RUNPOD_CONFIG", REMOTE_DIR / "runpod.json"))
        if not path.is_file():
            raise ValueError(f"Create {path} using src/remote/runpod.example.json; see docs/runpod.md")
        return cls(**json.loads(path.read_text()))

    def validate(self):
        if sys.version_info[:2] != (3, 12):
            raise ValueError("Use Python 3.12 for both the local SDK client and worker image")
        for name in ("image", "volume_id", "datacenter", "gpu"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or "YOUR_" in value or "<" in value:
                raise ValueError(f"Configure a real {name} in runpod.json")
        if not re.fullmatch(r"[A-Za-z0-9-]+", self.datacenter):
            raise ValueError("Invalid Runpod datacenter")
        if self.compute_datacenter is not None and not re.fullmatch(r"[A-Za-z0-9-]+", self.compute_datacenter):
            raise ValueError("Invalid compute datacenter")
        price = self.max_gpu_price_per_hour
        if type(price) not in (int, float) or not math.isfinite(price) or price <= 0:
            raise ValueError("max_gpu_price_per_hour must be a positive finite number")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.prefix):
            raise ValueError("prefix must be a single directory name")
        if ":" not in self.image.rsplit("/", 1)[-1] or self.image.endswith(":latest"):
            raise ValueError("Use the versioned image returned by build_worker_image(), not latest")
        for name in ("startup_timeout", "job_timeout", "container_disk_gb", "max_placement_attempts"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        required = ("RUNPOD_API_KEY", "RUNPOD_S3_ACCESS_KEY_ID", "RUNPOD_S3_SECRET_ACCESS_KEY")
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise ValueError("Set local environment variables: " + ", ".join(missing))

    @property
    def s3_endpoint(self):
        return f"https://s3api-{self.datacenter.lower()}.runpod.io"
