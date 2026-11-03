"""Immutable, hash-checked bundles shared through the network volume."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile

from .config import ROOT

CORE_FILES = (
    "options.py", "stroke_plan.py", "painting.py", "painting_optimization.py",
    "paint_utils3.py", "brush_stroke.py", "param2stroke.py", "my_tensorboard.py",
    "losses/clip_loss.py", "stroke_renderer.py", "foreground_planning.py", "project_paths.py",
)
CACHE_FILES = (
    "brush_tip_to_table_calib.json", "cached_H_canvas.pkl", "cached_H_coord.pkl",
    "stroke_settings_during_library.json", "param2stroke_settings.json", "param2img.pt",
)


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root, name):
    rel = PurePosixPath(name)
    if not name or rel.is_absolute() or ".." in rel.parts or "\\" in name or str(rel) != name:
        raise ValueError(f"Unsafe artifact path: {name!r}")
    root = Path(root).resolve()
    path = root.joinpath(*rel.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Artifact escapes its directory: {name!r}")
    return path


def make_manifest(root):
    root = Path(root)
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not supported in bundles: {path}")
        if path.is_file() and path != root / "manifest.json":
            files[path.relative_to(root).as_posix()] = sha256(path)
    return {"version": 1, "id": hashlib.sha256(canonical(files)).hexdigest(), "files": files}


def validate_manifest(manifest):
    if (manifest.get("version") != 1 or not isinstance(manifest.get("files"), dict)
            or manifest.get("id") != hashlib.sha256(canonical(manifest["files"])).hexdigest()):
        raise ValueError("Invalid bundle manifest")
    for name, digest in manifest["files"].items():
        safe_path(Path("/tmp/frida-artifact-validation"), name)
        if name == "manifest.json" or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Invalid bundle file entry")
        int(digest, 16)


def verify_bundle(root, manifest):
    validate_manifest(manifest)
    for name, digest in manifest["files"].items():
        if sha256(safe_path(root, name)) != digest:
            raise ValueError(f"Artifact checksum mismatch: {name}")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical(data))
    temporary.replace(path)


def code_version():
    paths = [ROOT / "LICENSE"] + [ROOT / "src" / name for name in CORE_FILES]
    paths += sorted((ROOT / "src/remote").glob("*.py"))
    paths += [ROOT / "src/remote" / name for name in
              ("Dockerfile", "requirements-client.txt", "requirements-worker.txt")]
    return hashlib.sha256(canonical({p.relative_to(ROOT).as_posix(): sha256(p) for p in paths})).hexdigest()


def copy_cache(source, destination, *, training=False):
    source, destination = Path(source), Path(destination)
    required = ["stroke_settings_during_library.json"]
    if not training:
        required += ["param2img.pt", "param2stroke_settings.json"]
    for name in required:
        if not (source / name).is_file():
            raise ValueError(f"Missing cached file: {source / name}")
    library = source / "stroke_library"
    if not library.is_dir():
        raise ValueError(f"Missing stroke library: {library}")
    files = [source / name for name in CACHE_FILES if (source / name).is_file()]
    files += sorted(library.glob("stroke_intensities*.npy"))
    files += sorted(library.glob("stroke_parameters*.npy"))
    if training and not list(library.glob("stroke_intensities*.npy")):
        raise ValueError("Training needs recorded stroke intensities and parameters")
    (destination / "stroke_library").mkdir(parents=True, exist_ok=True)
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):
            raise ValueError(f"Cache file must be inside its cache directory: {path}")
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


class VolumeStore:
    def __init__(self, config=None, *, volume_id=None, datacenter=None):
        import boto3
        from botocore.config import Config
        import os
        import re

        if config is not None:
            volume_id, datacenter = config.volume_id, config.datacenter
        if not isinstance(datacenter, str) or not re.fullmatch(r"[A-Za-z0-9-]+", datacenter):
            raise ValueError("Invalid storage datacenter")
        if not isinstance(volume_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", volume_id):
            raise ValueError("Invalid storage volume")
        self.bucket = volume_id
        self.client = boto3.client(
            "s3", endpoint_url=f"https://s3api-{datacenter.lower()}.runpod.io", region_name=datacenter,
            aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"],
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                          request_checksum_calculation="when_required",
                          response_checksum_validation="when_required",
                          retries={"max_attempts": 4, "mode": "standard"}),
        )

    def _json(self, key):
        from botocore.exceptions import ClientError
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            with response["Body"] as stream:
                return json.load(stream)
        except ClientError as exc:
            if str(exc.response["Error"]["Code"]) in ("404", "NoSuchKey", "NotFound"):
                return None
            raise

    def upload(self, root, prefix):
        manifest = make_manifest(root)
        previous = self._json(f"{prefix}/manifest.json")
        if previous == manifest:
            return manifest
        if previous is not None:
            raise ValueError(f"Refusing to replace an immutable bundle: {prefix}")
        for name in manifest["files"]:
            self.client.upload_file(str(safe_path(root, name)), self.bucket, f"{prefix}/{name}")
        # Commit marker is last: workers never read a partially uploaded bundle.
        self.client.put_object(Bucket=self.bucket, Key=f"{prefix}/manifest.json", Body=canonical(manifest))
        return manifest

    def download(self, prefix, destination):
        destination = Path(destination)
        manifest = self._json(f"{prefix}/manifest.json")
        if manifest is None:
            raise FileNotFoundError(f"No completed result at {prefix}")
        validate_manifest(manifest)
        if destination.exists():
            verify_bundle(destination, manifest)
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".download-") as directory:
            staging = Path(directory) / "bundle"
            staging.mkdir()
            for name in manifest["files"]:
                path = safe_path(staging, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                self.client.download_file(self.bucket, f"{prefix}/{name}", str(path))
            verify_bundle(staging, manifest)
            write_json(staging / "manifest.json", manifest)
            staging.rename(destination)
        return destination
