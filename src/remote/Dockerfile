FROM python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONPATH=/opt/frida:/opt/frida/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/frida
COPY src/remote/requirements-client.txt src/remote/requirements-worker.txt src/remote/
RUN python -m pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install --no-cache-dir -r src/remote/requirements-worker.txt \
    && python -m pip check
# Download during the image build, before any paid GPU is provisioned.
RUN python -c 'import clip; clip.load("ViT-B/32", device="cpu", download_root="/opt/models/clip")'
COPY src/options.py src/stroke_plan.py src/painting.py src/painting_optimization.py \
     src/paint_utils3.py src/brush_stroke.py src/param2stroke.py src/my_tensorboard.py src/stroke_renderer.py src/foreground_planning.py src/project_paths.py src/
COPY src/losses/clip_loss.py src/losses/clip_loss.py
COPY src/remote/*.py src/remote/
COPY LICENSE LICENSE
COPY src/remote/Dockerfile src/remote/Dockerfile
RUN python -c 'import runpod_sdk_runtime.task.runner, painting_optimization; from remote.artifacts import code_version; print("FRIDA code:", code_version())'
# Runpod overrides CMD to start its authenticated task runtime.
CMD ["python", "-m", "runpod_sdk_runtime.task.runner"]
