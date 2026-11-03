"""One-time image preparation and a local readiness check; no GPU provisioning."""

import importlib.metadata
import getpass
import os
import re
import shutil
import subprocess
import sys
import tempfile

from .artifacts import CORE_FILES, code_version
from .config import ROOT, SDK_VERSION, RunpodConfig


def doctor(config_path=None):
    problems = []
    if sys.version_info[:2] != (3, 12):
        problems.append("Select the .venv-runpod Python 3.12 interpreter in your editor")
    for name, required in (("runpod", SDK_VERSION), ("boto3", None)):
        try:
            version = importlib.metadata.version(name)
            if required and version != required:
                problems.append(f"{name} must be {required}, found {version}")
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"Install {name} with src/remote/requirements-client.txt")
    try:
        config = RunpodConfig.load(config_path)
        config.validate()
    except (ValueError, TypeError, OSError) as exc:
        problems.append(str(exc))
    for problem in problems:
        print(f"Needs setup: {problem}")
    if not problems:
        print("Local configuration is ready. Cloud access/image availability have not been tested.")
    return problems


def build_worker_image(repository, *, push=False):
    """Build Linux/amd64 on Docker Desktop; optionally publish to your registry."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9./_-]+", repository) or "/" not in repository:
        raise ValueError("Pass a repository such as ghcr.io/your-name/frida-worker, without a tag")
    if shutil.disk_usage(ROOT).free < 25 * 1024**3:
        raise RuntimeError("Free at least 25 GiB before building the CUDA image (including temporary layers)")
    image = f"{repository}:{code_version()[:16]}"
    # Explicitly construct a small context. Even a registry/cloud builder never
    # receives local calibration, datasets, credentials, or unrelated repo files.
    from pathlib import Path
    with tempfile.TemporaryDirectory(prefix="frida-image-") as directory:
        context = Path(directory)
        paths = [ROOT / "LICENSE"] + [ROOT / "src" / name for name in CORE_FILES]
        paths += list((ROOT / "src/remote").glob("*.py"))
        paths += [ROOT / "src/remote" / name for name in
                  ("Dockerfile", "requirements-client.txt", "requirements-worker.txt")]
        for path in paths:
            destination = context / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        subprocess.run(["docker", "buildx", "build", "--platform", "linux/amd64", "--load",
                        "-f", "src/remote/Dockerfile", "-t", image, "."], cwd=context, check=True)
    # Test source imports and SDK runtime in the exact built image before push.
    subprocess.run(["docker", "run", "--rm", "--entrypoint", "python", image, "-c",
                    "import runpod_sdk_runtime.task.runner, painting_optimization; "
                    "from remote.artifacts import code_version; print(code_version())"], check=True)
    if push:
        subprocess.run(["docker", "push", image], check=True)
    print(f"Use this image in src/remote/runpod.json: {image}")
    return image


def prompt_credentials(*, api_only=False):
    """For the editor's Run button: ask only for missing keys, never save them."""
    names = ("RUNPOD_API_KEY",) if api_only else (
        "RUNPOD_API_KEY", "RUNPOD_S3_ACCESS_KEY_ID", "RUNPOD_S3_SECRET_ACCESS_KEY")
    for name in names:
        if not os.environ.get(name):
            os.environ[name] = getpass.getpass(f"{name} (hidden, not saved): ").strip()
