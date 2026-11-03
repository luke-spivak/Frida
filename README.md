# FRIDA for SO101

This checkout adapts [FRIDA](https://github.com/cmubig/Frida) for an SO101 painting
arm and an Innomaker camera. It supports calibration, stroke-library collection,
stroke-model training, image-based painting plans, and execution/resumption of
saved plans. Training and planning can run locally or on Runpod.

SO101 is the default and only physical robot backend. `--simulate` uses an
in-memory canvas and simulated robot without opening hardware. Simulation still
requires a matching saved stroke library and trained model.

## Repository layout

```text
src/             Painting, rendering, optimization, and robot code
  camera/        Innomaker capture, simulation, calibration, and recording
  losses/        Image-based CLIP objectives
  SO101/         Robot models, meshes, and so101_offsets.json
  caches/        Measured calibration, stroke libraries, and trained models
  remote/        Runpod client/worker, Dockerfile, requirements, and runpod.json
scripts/         Calibration, diagnostics, experiments, and runpod_workflow.py
tests/           Automated regression tests
targets/         Input images and palette images
outputs/        Captures, painting_log, recordings, and job/experiment results
docs/           Runpod instructions and validation history
```

## Environment and setup

Use the existing `frida-lerobot` environment for the robot and local painting.
The hardware path uses LeRobot's SO101 follower/Feetech support, RoboPlan,
Pinocchio, and Viser alongside the painting and camera dependencies in
`requirements.txt`. The dependency versions in that file and the Linux export
in `environment.yml` are inherited pins, not a lockfile for the current SO101
Mac environment; they do not install the complete robot stack.

Run commands below from the repository root. Paths passed on the command line
are relative to the working directory; built-in resource paths resolve from the
source files.

Supply your actual materials JSON and the cache matching the physical brush and
calibration. The materials JSON specifies canvas geometry, station positions,
and stroke limits; positions are in meters relative to the robot. The old
`materials.json` and `src/caches/cache_6_6_cvpr` defaults are absent in this
checkout, so pass `--materials_json` and `--cache_dir` explicitly.

Follow the [SO101 operating and calibration instructions](src/SO101/README.md)
for brush geometry, motor calibration, canvas height, and station setup. Measured
calibration, stroke libraries, and trained models are stored together in the
selected cache.

## Plan and paint

Create and save a plan in simulation, using an existing model:

```sh
python src/paint.py --materials_json my_materials.json \
  --cache_dir src/caches/so101_mvp --use_cache --dont_retrain_stroke_model \
  --simulate --objective l2 --objective_data targets/black_bear.png --objective_weight 1
```

The default plan destination is `cache_dir/stroke_plan.json`; override it with
`--save_plan`. Simulation leaves all strokes pending.

Execute or resume a saved plan with the matching physical setup:

```sh
python src/paint.py --materials_json my_materials.json \
  --cache_dir src/caches/so101_mvp --use_cache \
  --load_plan src/caches/so101_mvp/stroke_plan.json
```

Loading a plan validates its setup and calibration/model fingerprints before
connecting to the robot, skips optimization and training, and resumes remaining
strokes. The startup preview lets you inspect the canvas and palette before
execution. `--dont_plan` loads the selected cache's default plan filename.

For a new physical painting, omit `--simulate` from the planning command.
`--num_adaptations` controls observation and replanning during that run. Without
`--use_cache`, startup can collect calibration and a stroke library; without
`--dont_retrain_stroke_model`, it can train the stroke model. Use those operations
deliberately when preparing a new brush/setup.

Use `--help` for the full option list. Relevant controls include `--num_strokes`,
`--init_optim_iter`, `--optim_iter`, `--render_height`, and
`--how_often_to_get_paint`. `--ink` supports a marker or brush pen without paint
loading or washing.

## Image objectives and palettes

The retained objectives compare a rendered painting with a reference image:

- `l2`: mean squared pixel error.
- `clip_conv_loss`: similarity of intermediate CLIP visual features.
- `clip_fc_loss`: similarity of CLIP image embeddings.

Local planning can combine objectives. Supply one image and weight per objective:

```sh
python src/paint.py --simulate --use_cache --dont_retrain_stroke_model \
  --materials_json my_materials.json --cache_dir src/caches/so101_mvp \
  --objective clip_conv_loss l2 \
  --objective_data targets/black_bear.png targets/black_bear.png \
  --objective_weight 0.2 1.0
```

Use `--use_colors_from` with a palette image and `--n_colors` to constrain paint
colors. `--foreground_mode` plans a black silhouette on white, treating
`--num_strokes` as a maximum budget. It requires one image objective and a fixed
black palette, such as `--use_colors_from targets/palettes/black.png --n_colors 1`.

## Runpod

Use the separate Python 3.12 Runpod client environment:

```sh
.venv-runpod/bin/python scripts/runpod_workflow.py --help
```

The workflow's default action provisions a paid job. See [Runpod setup and
workflows](docs/runpod.md) for training, planning, downloading model bundles, and
building the worker image. Rebuild and publish the worker image after changes
to its bundled source or dependencies, and update `src/remote/runpod.json` with
the returned tag.

## Camera, recording, and diagnostics

The physical camera uses the Innomaker through OpenCV's `cv2.VideoCapture(0)`.
Connect it and allow camera access for the Python process.
`scripts/innomaker_test.py` checks capture and the calibrated canvas view.
Physical painting records overhead video and events by default; use
`--no_recording` to disable this. Simulation never records video.

Examples of offline previews and checks:

```sh
python scripts/paint_repeatability.py
python scripts/preview_so101_offsets.py --help
python -m pytest -q
```

The first command previews the repeatability experiment; physical execution
requires its explicit execution option. See the SO101 instructions for motion
diagnostics and paint loading, washing, and repeatability experiments.

View optimization logs with TensorBoard, if installed:

```sh
tensorboard --logdir outputs/painting_log
```

## Attribution

FRIDA (a Framework and Robotics Initiative for Developing Arts) was developed at
The Robotics Institute, Carnegie Mellon University, and maintained upstream by
Peter Schaldenbrand. See [FRIDA: A Collaborative Robot Painter with a
Differentiable, Real2Sim2Real Simulated Planning Environment](https://arxiv.org/abs/2210.00664)
for the original research and [the upstream repository](https://github.com/cmubig/Frida)
for its broader research history. This checkout supports the workflows described
above.

Upstream contributors acknowledged by the project include Sunyu Wang, Jia Chen
Xu, Heera Sekhr, Jesse Ding, Vihaan Misra, Tanmay Shankar, and Kevin Zhang.
See [LICENSE](LICENSE) for licensing terms.
