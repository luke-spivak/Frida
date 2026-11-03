# Runpod painting and brush-model training

## Run targets/images.jpeg

Open `scripts/runpod_workflow.py`, select `.venv-runpod/bin/python`, and press **Run**.
Or run `.venv-runpod/bin/python scripts/runpod_workflow.py` from the repository root.
The script prompts for missing Runpod and S3 credentials once, keeps them in
memory for that session, and starts a paid 400-stroke, 20-iteration CLIP test.
The robot's `frida-lerobot` environment stays separate.

The existing volume `u8lwd7nc27` in `EUR-IS-1` is now only an S3 file store.
GPU Pods do **not** mount it and can run in any supported region. Inputs are
uploaded before provisioning; the worker downloads them onto its temporary disk,
runs the original FRIDA code, and uploads completed results before terminating.
The Mac downloads those results after GPU cleanup. CLIP weights are baked into
the image, so the GPU does not need to download them from OpenAI on startup.

No new volume, account, registry, API key, or local package installation is needed
for an existing setup. Keep the existing storage datacenter and volume settings.

## GPU selection and capacity failures

`src/remote/runpod.json` now separates storage from GPU placement:

```json
{
  "volume_id": "u8lwd7nc27",
  "datacenter": "EUR-IS-1",
  "gpu": "auto",
  "compute_datacenter": null,
  "max_gpu_price_per_hour": 2.0,
  "max_placement_attempts": 12
}
```

These are selected fields, not a replacement for the complete configuration;
retain its versioned `image` and other settings. `datacenter` is the **storage**
region. `compute_datacenter: null` allows GPUs in any region. Set it explicitly
only to restrict compute. `gpu: "auto"` considers SDK-supported NVIDIA GPUs with
at least 24 GB VRAM; an exact GPU name restricts the search to that type.

The catalog is queried for one Secure Cloud Pod and a minimum CUDA host version
of 12.8. Compatible host versions advertised by the live catalog (such as 13.2)
are registered in the pinned SDK before deployment; its old built-in version list
must not exclude newer compatible drivers. Candidates above the $2/hour catalog
estimate, or without a known price, are excluded. Higher stock is preferred,
then 5090/4090/RTX 6000 Ada/A100 types, then other supported GPUs. Catalog prices
are estimates, not an enforced maximum on the actual deployed Pod bill.

At most 12 GPU/region placements are attempted. Only an explicit deployment
capacity rejection before Pod creation advances to the next candidate. Timeouts,
authentication errors, ambiguous startup failures and worker failures stop the
run; they never automatically start another paid job. A complete capacity
failure lets you press Enter to refresh and retry with the same credentials.

For a read-only availability list, use `--availability` or `ACTION = "availability"`.
This only prompts for the Runpod API key. `ACTION = "check"` checks local setup.

## Fresh account or image setup

Create one network volume in a datacenter supporting the
[Runpod S3 API](https://docs.runpod.io/storage/s3-api), and create Runpod API and S3
credentials. Its GPU stock does not constrain compute. Put non-secret settings
in `src/remote/runpod.json`, following `src/remote/runpod.example.json`. Never put keys in source,
Dockerfiles, or configuration files. You may supply them through environment
variables instead of the hidden prompts.

Use Python **3.12** and `src/remote/requirements-client.txt` locally. Docker Desktop
builds the Linux/amd64 image; leave at least 25 GiB free for layers. Authenticate
to Docker Hub/GHCR, then start Python from the repository root with
`PYTHONPATH=src .venv-runpod/bin/python` and build and publish with:

```python
from remote.setup import build_worker_image
image = build_worker_image("your-name/frida-worker", push=True)
```

Set the returned versioned tag in `src/remote/runpod.json`. Private images require
`registry_auth` configured in Runpod. Rebuild after worker source/dependency
changes: both client and worker enforce a matching source fingerprint.

The worker source layout has changed: rebuild and publish the worker image after
this reorganization, then update `src/remote/runpod.json` with the returned tag.
Existing plans and downloaded model bundles retain their original data format.

The SDK is pinned to `2.0.0rc1` and the runtime to an official Git commit.
The client registers selected live-catalog regions missing from the SDK's old
datacenter enum in memory; installed package files are not changed.

For the Python API examples below, start Python from the repository root with
`PYTHONPATH=src .venv-runpod/bin/python`. The script entry point sets its import
path automatically and can also be launched from another working directory.

## Everyday use

Set `ACTION` in `scripts/runpod_workflow.py` to `"plan"`, `"train"`, or `"train_and_plan"`
and press **Run**. The example already points at `src/caches/so101_mvp`,
`my_materials.json`, and `targets/images.jpeg`. These actions start a paid task.
The example's `plan` action starts with 400 strokes and **20 iterations** to
measure speed. Increase it to 400 iterations for a full optimization after the
first successful run. The Python helper's default remains 400 iterations.

Or call the helpers from Python:

```python
from remote import plan_remotely, train_brush_remotely, train_and_plan_remotely

# Use your existing SO-101 model. Start small to measure actual speed and quality.
plan = plan_remotely(
    target="targets/images.jpeg",
    cache_dir="src/caches/so101_mvp",
    materials_json="my_materials.json",
    objective="clip_conv_loss",
    num_strokes=100,
    iterations=100,
)
print(plan.preview_path, plan.plan_path, plan.model.cache_dir)

# Train a NEW version from already-recorded physical strokes.
model = train_brush_remotely(cache_dir="src/caches/so101_mvp")
plan = plan_remotely(target="targets/images.jpeg", model=model)

# Or train and plan in the SAME Pod, avoiding a second startup.
plan = train_and_plan_remotely(
    target="targets/images.jpeg",
    cache_dir="src/caches/so101_mvp",
    training={"max_iterations": 5000, "patience": 200},
    objective="clip_conv_loss",
)
```

Planning defaults match FRIDA's 400 strokes, 400 optimization iterations, render
height 256, and 12 palette colors. Supported remote objectives are `clip_conv_loss`,
`clip_fc_loss`, and `l2`. The target must be an image file. Local planning
supports the same objectives and can combine them; remote planning selects one
objective per job.
`num_strokes` must be a perfect square because the existing initializer uses a
square grid. `num_augs` affects the CLIP-FC objective; CLIP-convolution keeps
its existing fixed 10 augmentations. Lower iteration/stroke counts trade quality
for runtime; moving to an A100 does not guarantee a one-minute result.

This plans on a **blank canvas**, just like the simulation. Physical stroke data
collection and executing the resulting plan remain local. It does not provide
live camera feedback or repeated remote adaptations during robot execution.

## Fixed palettes

The planning helpers accept either explicit colors or a palette-source image.
Both routes pass an input palette into FRIDA's existing optimizer; the palette
centers are not relearned from the painting.

```python
# Use four colors extracted from the target, fixed for this optimization.
plan = plan_remotely(
    target="targets/kuka.jpeg",
    cache_dir="src/caches/so101_mvp",
    materials_json="my_materials.json",
    use_colors_from="targets/kuka.jpeg",
    n_colors=4,
    objective="l2",  # or "clip_conv_loss"
    num_strokes=144,
    iterations=400,
)

# Alternatively, specify exact paint colors. n_colors is inferred if omitted.
plan = plan_remotely(
    target="targets/kuka.jpeg",
    cache_dir="src/caches/so101_mvp",
    materials_json="my_materials.json",
    palette=["#C86E32", "#000000"],
    objective="l2",
    num_strokes=144,
    iterations=400,
)
```

Explicit colors can be `#RRGGBB` strings or RGB triples in `[0, 1]`. An explicit
`n_colors` must match their count. `palette` and `use_colors_from` are mutually
exclusive and cannot be combined with ink mode. Both options also work with
`train_and_plan_remotely`.

Image palettes use the same 256 × 256 resize and RGB k-means as local
`paint.py --use_colors_from`. All image pixels participate, including white
background; two colors extracted from the whole KUKA photo produce near-white
and brown, not orange and black. Use a swatch image or explicit RGB values when
the exact colors matter. The source image is uploaded with the job inputs.

The Run-button workflow also accepts `--use_colors_from swatches.png --n_colors 4`
or `--palette '#C86E32' '#000000'`, retaining its existing target and iteration
defaults. Neither flag starts an additional job by itself.

`input_palette.json` records the supplied/extracted colors; `palette.json` and
the final stroke plan retain them. A supplied **one-color palette** initializes
every stroke to that color before the first render and keeps it exact throughout
optimization, including pure black or white. Geometry still optimizes normally.
For black paint, use `use_colors_from="targets/palettes/black.png", n_colors=1`
(CLI: `--use_colors_from targets/palettes/black.png --n_colors 1`) or
`palette=["#000000"]`. Extracting one color from a black subject on a white
background would instead produce the image's average gray.

With multiple input colors, the existing schedule is unchanged: continuous
stroke colors may drift and be clamped between scheduled palette snaps, then
snap to the input palette at the end. Learned palettes also retain their
previous behavior.

## Renderer performance controls

Planning renders strokes in batches of 64 by default. Coordinate grids and
constant normalization transforms are cached on the rendering device; learned
stroke images and their position/rotation transforms are evaluated in batches.
FRIDA still composites strokes one at a time in their current order. The
optimizer's color sorting, early shuffling, palette updates, learning rates,
CLIP objective, and ten CLIP-convolution augmentations are unchanged.

The Python planning helpers accept `render_batch_size=64` and `cpu_threads=N`.
Set `render_batch_size=0` to use the original per-stroke renderer for comparison.
`cpu_threads` controls PyTorch's CPU thread pool during planning, including for
`train_and_plan`; it does not change brush-model training. Leaving it as `None`
retains PyTorch's environment default. Local FRIDA accepts `--render_batch_size`.
Batching preserves the equations and stroke order, with small floating-point
differences; optimization trajectories need not be bit-for-bit identical.

The live A40 comparison for `targets/images.jpeg` (400 strokes, 20 iterations,
CLIP-convolution, 256 × 330) measured **222.54 s** for the original renderer,
**46.91 s** for the original renderer with four CPU threads, and **9.40 s** for
batched rendering: **23.66× faster planning**. These are worker planning times
with CLIP already loaded, excluding Pod startup and file transfers. The batched
CPU-thread sweep ranged from 0.343 to 0.387 s per iteration for 1/2/4/8/48
threads; reducing threads offered no clear improvement after batching. Leave
`cpu_threads=None` for the normal workflow. The benchmark's PyTorch default
was 48 threads. See [the measured results and correctness checks](runpod-validation.md).

A subsequent normal **400-iteration A40 run** took **2m44s for planning**,
including initial CLIP loading, and **6m21s end to end**, including 3m12s of
startup. Its preview and exact timing artifacts are linked in the validation
report. This full run leaves the example workflow's 20-iteration default intact.

Changing only the objective to `l2` in the same 400-iteration A40 setup measured
**2m18s planning** and **5m26s total**. It reduced exported pixel error by 82% and
filled the target's large red region much more completely. See the validation
report for previews, coverage metrics and the remaining stroke/color limits.
This comparison does not change the helper or workflow's CLIP default.

## Files and matching model versions

Results go to `outputs/runpod/<job-id>/`:

- `stroke_plan.json`, `preview.png`, `palette.json` for planning.
- `cache/param2img.pt` and `cache/param2stroke_settings.json` for the model.
- The exact calibration files, recorded stroke library, and `materials.json`.
- `cache/training_metrics.json` and PNG comparison plots for training.
- `result.json` with source/model versions, device, settings, and worker timings.
- `manifest.json` with SHA-256 hashes for all returned files.

Your source cache is never overwritten. A plan is checked against its downloaded
model, calibration, and materials using FRIDA's existing `StrokePlan.check_setup`.
Keep these together. You can load a previous result with:

```python
from remote import ModelResult, plan_remotely
model = ModelResult.load("outputs/runpod/<previous-job-id>")
plan = plan_remotely(target="targets/images.jpeg", model=model)
```

To preview a downloaded plan through the existing local simulation, use your
normal **FRIDA robot environment**, not `.venv-runpod`:

```sh
python src/paint.py --robot so101 --simulate --use_cache --dont_retrain_stroke_model \
  --cache_dir outputs/runpod/JOB_ID/cache \
  --materials_json outputs/runpod/JOB_ID/materials.json \
  --load_plan outputs/runpod/JOB_ID/stroke_plan.json
```

For an ink job include `--ink`. For a different robot use the recorded robot name.
Do not switch to another model/cache when executing the plan. Physical execution
still goes through the existing local calibration checks and ready prompt.

## Runtime, billing, and interrupted jobs

Runpod's S3 API does not support presigned URLs. The temporary worker receives
the existing S3 credentials through its runtime environment to transfer bundles.
The Runpod API key is also passed for the SDK's self-termination watchdog.
Credentials are never written into the public image, task request, local receipt,
or result bundle. They remain available to the Pod for its lifetime.

Uploads and downloads verify SHA-256 manifests. Completed output manifests are
published last. The worker uses temporary local disk, with no network-volume
mount or region-dependent filesystem requirement. All three operations—plan,
train, and train-and-plan—use this transfer path. Recorded robot data stays
unchanged locally.

The client requests termination on success/error/cancellation, then downloads
completed results. Defaults allow 15 minutes for startup and 30 minutes for work;
these are deadlines, not runtime or spending promises. Paid time includes image
pull/startup, S3 transfers on the Pod, computation, and termination delays. The
existing storage volume continues to have its separate charge. Keep the Mac
awake until completion. See [Pod pricing](https://docs.runpod.io/pods/pricing).

Receipts in `outputs/runpod/<job-id>.json` record every placement attempt,
selected GPU and compute region, storage region, Pod ID once spawn returns,
transfer timings, and cleanup status. `result.json` records the worker's actual
GPU and training/planning timings.

If cleanup is unconfirmed, inspect the Runpod console for `task-frida_gpu_job-*`
Pods. A receipt with `pod_created: false` records a definite rejection before
creation. A lost connection or hard-killed client can prevent cleanup confirmation.

If only the local download failed, recover without another GPU:

```python
from remote.client import recover_results
directory = recover_results("outputs/runpod/<job-id>.json")
```

On a worker computation failure, partial diagnostics are uploaded best-effort to
`frida/jobs/<job-id>/failed`. A worker killed before its result upload completes
cannot preserve its temporary files. Uploaded inputs, model versions and completed
results remain on the storage volume until intentionally deleted.

## Validation

`tests/test_remote.py` and `tests/test_runpod_workflow.py` cover real CPU
training/planning, S3 bundle transfers, model reuse, cross-region SDK payloads,
capacity-only fallback, failure cleanup, and one-session credential handling.
The automated tests provision no cloud resources. A separate live A40 run has
verified cross-region S3, 400-stroke/20-iteration CUDA CLIP planning, result
retrieval and Pod deletion; see [validation and measured timings](runpod-validation.md).
## Black silhouettes on white

For a black silhouette such as `targets/black_bear.png`, pass
`foreground_mode=True` to `plan_remotely`, with `n_colors=1` and
`use_colors_from="targets/palettes/black.png"`. This is opt-in; ordinary
painting jobs keep their existing behavior. Local planning accepts
`--foreground_mode` with the same single black palette and a white canvas.

This mode initializes real brush footprints inside the target, proposes
replacements in uncovered foreground every 50 iterations, and removes
unhelpful or out-of-bounds strokes. The requested `num_strokes` is a maximum
budget; `foreground_diagnostics.json` records initialization, relocation,
removal, and final counts. A retained stroke may place at most 2% of its
predicted paint outside the silhouette. This checks the entire learned mask,
including brush width; no image masking or synthetic opacity is exported.

L2 remains the image objective, with an additional background pixel penalty
of `foreground_background_weight=4.0`. CLIP also gets a full-canvas L2 term
of `foreground_reconstruction_weight=1.0`. Thus a CLIP run in this mode is
CLIP plus reconstruction/background guidance, not the earlier pure CLIP loss.
The stroke model, physical limits, compositing, color ordering and RMSprop
schedule stay the same. Surviving strokes are exported normally and can be
loaded with `paint.py --load_plan`; no foreground mask is needed to execute
the saved plan. Silhouette planning currently requires a white starting
canvas and does not support feedback optimization on an already painted canvas.

