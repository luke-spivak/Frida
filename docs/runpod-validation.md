# Runpod implementation validation

Historical commands below retain the paths used during those runs. Current
commands and the relocated `scripts/runpod_workflow.py`, `src/remote/`, and
`tests/` paths are documented in [Runpod setup](runpod.md).

## Black bear, black held from the first render — 2026-10-28

The six bear jobs were repeated with **100, 196 and 400 strokes, 400 iterations,
L2/CLIP-convolution and one A40 per job**. Each now uses
`use_colors_from="targets/palettes/black.png", n_colors=1`: a solid-black swatch,
not the average color of the bear image and its white background. Paint mode,
target, model, calibration, materials, seed zero, 397 × 256 rendering, batch
size 64, default CPU threads, fill weight zero and all other settings match the
previous six jobs.

Only supplied one-color palette handling changed in `painting_optimization.py`:
copy the input color into every stroke before the first render, set the color
optimizer's learning rate to zero, and restore the color immediately after
generic validation. The latter is necessary because validation clamps black to
0.02. Geometry optimization, renderer, initializer, losses and the sorting/
shuffling schedule are unchanged. The other nine core source hashes match the
pre-edit snapshot. Explicit one-color palettes get the same behavior; learned,
multi-color and ink paths retain their previous behavior.

| Objective | Strokes | Planning | Total client elapsed | New RGB MSE | Silhouette IoU: previous → new |
| --- | ---: | ---: | ---: | ---: | ---: |
| L2 | 100 | 42.56 s | 108.66 s (1m49s) | 0.09230 | 78.53% → 73.63% |
| L2 | 196 | 71.11 s | 271.89 s (4m32s) | 0.14575 | 68.88% → 65.83% |
| L2 | 400 | 132.78 s | 345.14 s (5m45s) | 0.20273 | 61.57% → 58.39% |
| CLIP | 100 | 67.52 s | 294.09 s (4m54s) | 0.31055 | 36.40% → 34.31% |
| CLIP | 196 | 87.38 s | 316.26 s (5m16s) | 0.42205 | 36.04% → 37.18% |
| CLIP | 400 | 164.86 s | 362.52 s (6m03s) | 0.60873 | 29.42% → 31.21% |

Holding black from the start **did not resolve the stray background strokes**.
The new L2/100 is the closest of these new results, but all three L2 runs have
lower silhouette overlap and higher pixel error than their earlier counterparts.
CLIP improves slightly at 196/400 and worsens at 100; its background artifacts
remain substantial. The comparison therefore does not support palette drift as
the sole explanation of the poor results. No further quality changes or extra
GPU experiments were made. Metrics use the same masks and definitions as below;
these are single seeded observations on separate A40 hosts.

All six Pods ran in CA-MTL-1, each reporting $0.59/hour and one A40, at most two
concurrently. All succeeded on their first attempt without result-poll retries.
Startup ranged from 22.98 to 190.05 seconds; the unusually fast L2/100 startup
accounts for its shorter total elapsed time. The six-run batch took
**1026.45 seconds (17m06s)** from first launch through final download.

Validation:

- **76 focused tests passed**, including public API/worker transfer checks that
  assert exact black before every rendering call for single-color image and
  explicit palettes, in both planning and train-and-plan.
- A separate 40-iteration real-model CPU check asserts exact black at every loss
  render, unchanged initial geometry, and continued geometry optimization.
  Learned-color, multi-color and ink cases produced bitwise identical CPU
  parameters/palettes to the pre-edit optimizer after six iterations.
- All **282 output-file checksums** passed; all **54 intermediate plan PNGs**
  have identical RGB channels. Final palettes and every saved stroke are exactly
  `[0,0,0]`, in palette slot zero. Uploaded bear/swatch inputs match local files.
- All six plans match both downloaded and current local calibration, reproduce
  their GPU PNGs on CPU within 0.00410 maximum RGB error including quantization,
  and pass the actual `paint.py --simulate --load_plan` path without changing
  any saved stroke. Physical hardware motion was not tested.
- All six Pods were independently confirmed deleted; no FRIDA Pods remained.
  The temporary credential process was closed and its handoff helpers removed.

Artifacts under `outputs/runpod-black-bear-locked/` include `comparison.png`,
`before-after.png`, `comparison.json`, `cleanup.json`, per-job outputs/runtime
records and `simulation-checks/`. `reproduce/` preserves the exact source diff,
pre-edit source, tests/regression results, worker build metadata, launchers,
input checks and comparison scripts without credentials.

```text
L2 100:   c3555ca1c4274bad8d334c6318dd5604  Pod 9crbrup7u16txu
L2 196:   8dc32a84c89743808f95eb63f2519fff  Pod pzjanhmffsz3ks
L2 400:   908be9d2569b4d2aa05eb65dd036c22e  Pod xue8fma8fk4ad0
CLIP 100: a7208d43b3a049aa949291d397981d7d  Pod looreedbhgshbf
CLIP 196: 0ebbc93705d0498ab57f78f0fbdf449e  Pod lo3dlmm5otqgkj
CLIP 400: e0c940614f454ada9cf79362181e8d0b  Pod k0o4myrvig2hyw
Image: thaigrapecucumber/frida-worker:5ff1b48e9ca78ed1
Manifest: sha256:745c78d698c537e148b00642d34bf6d472d266b07f22779190b5a13aa8708f76
Source: 5ff1b48e9ca78ed11ebb9293e2f88513bfc494b4a22dcdefe006ef657624d8dc
```

## Black bear, one black paint color — 2026-10-27

Six separate A40 jobs completed for `targets/black_bear.png`, using L2 and
CLIP-convolution at **100, 196 and 400 strokes**, each with **400 iterations**.
The user selected 196 instead of 200 to preserve the existing square-grid
initializer. Each request supplied `palette=["#000000"]`, `n_colors=1` and
`ink=False`: ordinary paint mode with black in palette slot zero. Seed zero,
batch size 64, default CPU threads, fill weight zero, the current SO-101 model
and 397 × 256 rendering were retained. No worker rebuild, optimizer, initializer,
renderer or workflow-default change was needed; all ten core source hashes and
the worker fingerprint match the pre-run snapshot.

| Objective | Strokes | Planning | Total client elapsed | RGB MSE | Silhouette IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| L2 | 100 | 42.55 s | 255.38 s (4m15s) | 0.07345 | **78.53%** |
| L2 | 196 | 81.67 s | 272.32 s (4m32s) | 0.12726 | 68.88% |
| L2 | 400 | 136.96 s | 325.35 s (5m25s) | 0.17747 | 61.57% |
| CLIP | 100 | 64.82 s | 275.45 s (4m35s) | 0.29271 | 36.40% |
| CLIP | 196 | 106.72 s | 283.11 s (4m43s) | 0.44479 | 36.04% |
| CLIP | 400 | 180.46 s | 401.18 s (6m41s) | 0.65570 | 29.42% |

Silhouette IoU thresholds both target and exported preview at mean RGB below
0.5; higher means greater overlap. The 100-stroke L2 result is closest overall,
but still has stray background marks and gaps within the bear. More L2 strokes
fill the bear more completely while increasing background marks: the fraction
of background painted dark rises from 9.05% to 18.16% to 25.47%. CLIP performs
worse at every count and paints 91.77% of the background dark at 400 strokes.
These are single seeded observations, not a general stroke-count scaling law.

The test exposes a limitation of the existing input-palette behavior: colors
are freely optimized early, then snapped to the input palette starting at
zero-based iteration 210. Thus this is a fixed **final palette**, not a test
with every stroke constrained to black throughout optimization. The L2/400
intermediate preview has RGB MSE 0.0404 at logged step 200, with many light or
colored background strokes; after the palette snapping starts it is 0.1875 at
step 250, finishing at 0.1775 with all-black strokes. This supports the color
schedule as a contributor to the unwanted background marks. No strict-black
ablation was run and no optimizer changes were made to improve the results.
The measurements for all six trajectories are saved in
`color-schedule-diagnostics.json`.

All jobs used one A40 in CA-MTL-1 at a reported $0.59/hour each, at most two
concurrently. Startup ranged from 137.25 to 182.84 seconds. All six succeeded
on their first attempt without transient result-poll retries. Wall time from
first launch through final download was **1106.14 seconds (18m26s)**.

All **282 output-file checksums** passed. Every plan has the requested stroke
count, only RGB `[0,0,0]`, palette index zero, and no completed/skipped strokes.
Uploaded targets match the local image; all jobs share the same source,
input/model bundle, calibration and materials. Every plan passes the setup
compatibility gate in physical mode using both its downloaded cache and the
current local SO-101 cache. CPU restoration reproduces the GPU PNG within
0.00410 maximum RGB error, including 8-bit export quantization.

Each of the six plans also passed the **actual `src/paint.py --simulate
--load_plan` entry point**, using a temporary copy of its cache and a separate
checkpoint. Optimization was skipped, palette slot zero was reported as black,
and the saved plan remained exactly equal, with no strokes marked completed.
The original downloaded bundles are unchanged. These checks validate software
integration; physical robot motion, reachability, paint deposition and camera
feedback were not tested.

All six Pods were independently confirmed deleted, no FRIDA Pods remained,
and the in-memory credential process was closed and its handoff helpers removed.
Artifacts are under `outputs/runpod-black-bear/`: the comparison PNG/JSON,
per-job plans/previews/diagnostics and runtime records, `simulation-checks/`,
`cleanup.json`, and credential-free reproduction scripts and metadata.

```text
L2 100:   fc4e127994fa4726be02688904f78c0c  Pod ciih00s6d12i38
L2 196:   d18f089b8176457abd52d215324d7044  Pod 1wpnrtj1j18zfr
L2 400:   dee125e9a42140e185fb4cec6d31c5d6  Pod z2ts1zd27cwwxv
CLIP 100: d27cedab3de840f6b1aec3adef35a3fb  Pod erwu4xbx0a4nzt
CLIP 196: 134913b8fd25437497543b66fde816ca  Pod gtochwdqhc9jvr
CLIP 400: eb79d5b412dc4f5fa5b6d4190f1c1352  Pod 8o9iri1okafon0
Target SHA-256: ce028a27e1e98030b31ce838e0e97b4b8619bf0576fdb4cd2f00578ddda8d499
Image: thaigrapecucumber/frida-worker:de1b6a2a28514071
```

## KUKA, four fixed colors from the target — 2026-10-27

The remote planning helpers now accept `use_colors_from` or explicit hex/RGB
`palette` values. Two separate A40 jobs used **144 strokes, 400 iterations and
four fixed colors extracted from `targets/kuka.jpeg`**, with the same target as
the painting objective. Extraction uses FRIDA's existing 256 × 256 RGB k-means.
The resulting palette was identical in both jobs:
`#FFFEFE`, `#C36121`, `#31231E`, `#D69A71` (rounded RGB display values).
The white image background contributes one paint color; white canvas is also
present. These are image-derived colors, not manually specified pure black and
orange.

The existing optimizer receives `color_palette` unchanged. Its stroke sorting,
early shuffling, color clamping/snapping schedule, renderer, loss functions and
learning rates are unchanged. Hashes of all ten core source files match the
snapshot taken before this feature. The new worker only exposes and transfers
the existing input-palette option. Seed zero, batch size 64, default CPU threads,
fill weight zero, current SO-101 model and 397 × 256 rendering were retained.

| Objective | Palette | Planning | Total client elapsed | Exported RGB MSE |
| --- | --- | ---: | ---: | ---: |
| L2 | Previous learned 2 colors | 55.01 s | 270.20 s | 0.03338 |
| L2 | New fixed 4 colors | **55.78 s** | **243.71 s (4m04s)** | **0.02556** |
| CLIP | Previous learned 2 colors | 73.68 s | 392.76 s | 0.14304 |
| CLIP | New fixed 4 colors | **75.69 s** | **264.92 s (4m25s)** | **0.13036** |

The L2 result preserves orange/dark regions and the robot silhouette much better
than the earlier brown silhouette. CLIP now uses appropriate colors but remains
fragmented, with substantial background marks. Pixel MSE decreased by 23.43%
for L2 and 8.86% for CLIP. This comparison changes **both palette source and
color count**, so it cannot isolate the effect of fixing the palette alone.
Planning times are single observations, not repeated performance benchmarks.

The foreground gap fractions stayed similar: L2 3.72% to 3.98%, CLIP 24.91%
to 24.35%. These measure combined stroke opacity below 5% over target pixels
whose minimum RGB channel is below 0.85; white strokes count as coverage.
Re-evaluation of exported PNGs with three paired CLIP augmentation seeds gave
0.63527 to 0.53101 for L2, and 0.60915 to 0.55621 for CLIP (lower is better).
These feature scores are not original training losses or human quality ratings.

Both new Pods ran in CA-MTL-1 at a reported $0.59/hour each, at most two
concurrently. CLIP startup took 128.93 seconds and L2 startup 149.03 seconds.
Both succeeded on their first attempt without result-poll retries. The unchanged
temporary retry wrapper covers only transient read-only result polling, never
submission or computation. Wall time from first launch through final download
was 287.36 seconds (4m47s).

All **94 new output-file checksums** passed. Both plans contain 144 strokes,
the same four input/final palette colors, and final stroke colors matching their
palette indices. Uploaded target and palette-source files match the local KUKA
SHA-256; model, calibration, input bundle and resized targets match the earlier
runs. Both Pods were independently confirmed deleted, no FRIDA Pods remained,
and the credential process was closed and handoff helpers removed.

Validation also passed **70 tests** across `test_remote.py` and
`test_runpod_workflow.py`, followed by **six end-to-end fixed-palette cases**
after adding four-color fixtures. These exercise explicit/image palettes,
planning and train-and-plan, exact color preservation, and public-client through
worker transfers. The rebuilt image passed its runtime import/source check.

Artifacts are under `outputs/runpod-kuka-fixed/`: `comparison.png`,
`comparison.json`, `cleanup.json`, per-objective plans/previews/palettes and
receipt/runtime records. `reproduce/` contains the exact runner, verification
script, source snapshots, verified uploaded-input records, safe Pod metadata
and logs. No credentials are saved there.

```text
L2:   7a9b6adc16c2466099efad47d6d139fa  Pod 17prli0dtcw7xb (deleted)
CLIP: 7e8085239e494035a7e9b67666a5c053  Pod 9e4wz23co9902g (deleted)
Image: thaigrapecucumber/frida-worker:de1b6a2a28514071
Manifest: sha256:b5ebfc819239c729c630f12c9d6a5a9bf8b6a4e764fb5c17ebcae9797751dbab
Source: de1b6a2a2851407112c919a6a836f22f2e6404225ed733e09acd8241bf13ed23
```

## KUKA, sparse two-color CLIP and L2 comparison — 2026-10-27

Five separate A40 jobs completed for `targets/kuka.jpeg`: CLIP-convolution at
64, 144 and 400 strokes, and L2 at 64 and 144 strokes. No existing KUKA baseline
was present locally, so the user explicitly authorized the additional 400-stroke
CLIP run. All jobs used **400 iterations, two learned paint colors, seed zero**,
batch size 64, default CPU threads, fill weight zero, the current retrained
SO-101 cache and 397 × 256 rendering. White canvas is additional to the two
paint colors. Palettes were learned independently using existing behavior.

| Objective | Strokes | Planning | Total client elapsed | Exported RGB MSE |
| --- | ---: | ---: | ---: | ---: |
| CLIP | 64 | 59.13 s | 246.17 s (4m06s) | 0.15581 |
| CLIP | 144 | 73.68 s | 392.76 s (6m33s) | 0.14304 |
| CLIP | 400 | 165.02 s | 356.34 s (5m56s) | 0.19006 |
| L2 | 64 | 34.50 s | 207.54 s (3m28s) | 0.05949 |
| L2 | 144 | 55.01 s | 270.20 s (4m30s) | 0.03338 |

The runs used at most two Pods concurrently, all in CA-MTL-1, each reporting
one NVIDIA A40, 9 vCPUs and $0.59/hour. Wall time from first job start to final
download was **881.19 s (14m41s)**. Startup ranged from 134.46 to 180.18 seconds.
The 144-stroke CLIP job spent 114.12 seconds downloading inputs, explaining why
its total elapsed exceeded the 400-stroke baseline despite faster planning.
These are single observations on separate hosts, not repeated benchmarks.

The 64-stroke CLIP result is sparse and loses much of the robot's structure;
144 strokes provide more structure but select two grayish colors. Neither
resembles a clean CLIPasso line sketch. The 400-stroke CLIP baseline preserves
a broad silhouette but paints the white background gray. L2 at 144 strokes
has the lowest pixel error and a more coherent silhouette, while merging the
orange/black robot into brown and retaining pale background marks. L2 at 64
strokes is faster but loses more of the silhouette.

Re-evaluation with the same CLIP objective and three fixed augmentation seeds
gave mean losses 0.59267, 0.60915 and 0.52205 for CLIP 64/144/400, respectively;
L2 64/144 scored 0.63129/0.63527. These are controlled evaluations of exported
PNG files, not original training-loss logs or human quality ratings. The
400-stroke CLIP image scores best under CLIP while L2/144 is closest in pixel
space, illustrating the objectives' different preferences. Foreground coverage
measurements and their precise masks are retained in `comparison.json`.

All **230 output files** passed checksum verification. Each plan contains the
requested number of strokes and exactly two palette colors. Worker source,
model weights, uploaded target hashes and all other planning settings match.
No optimizer, worker image or normal workflow defaults changed. The temporary
client reused the bounded read-only result-poll retry handling from Starship;
no retry or error events occurred in these five jobs.

All five Pods were independently confirmed deleted, no FRIDA Pods remained,
and the credential process was closed and handoff helpers removed.

Artifacts under `outputs/runpod-kuka/`:

- `comparison.png`: original previews together with the resized target.
- `comparison.json`: timings, palettes, coverage, pixel/feature metrics,
  source/input identity and independent cleanup evidence.
- Each objective/count directory contains its plan, palette, preview, matching
  cache, diagnostics and sibling receipt/runtime/verification records.
- `reproduce/` preserves the exact launcher and read-only verification script.

Successful jobs:

```text
CLIP 64:  49be8b136bea4fbd91a1cda29d5c7de2  Pod xf1k9krfj4dd1v
CLIP 144: 2ca240766bb04423a7c9d9a73b0c645b  Pod 6lsxf1100t62h4
CLIP 400: 945ad52c9ed94aabadd7c3febd407bf4  Pod 78ri7z7f2cqcxp
L2 64:    af505137b651427aa657d0290403c247  Pod 9s8nqwamtbcdo1
L2 144:   c958d87792454799a114e9bdff742e55  Pod aj3zt17ieayjcx
Target SHA-256: 93eb9e4f81d9cef8de9c5264be36758849cad9a73f0ef37510fb6abc7e88c2ad
```

Worker, source fingerprint, model/input version and weights match the
Michelangelo entry below: `thaigrapecucumber/frida-worker:63555d6034a0eee3`.

## Starship, separate L2 and CLIP runs with twelve colors — 2026-10-27

`targets/starship.jpg` completed as two separate A40 jobs with the retrained
SO-101 model: **400 strokes, 400 iterations and 12 colors**, seed zero,
batch size 64, default CPU threads and 397 × 256 rendering on the
21.6 × 13.9 cm canvas. The only planning-setting difference was the objective,
`l2` versus `clip_conv_loss`. Each run learned its own twelve-color palette.
The existing image matched local source; no rebuild or optimizer changes were
needed. Stroke ordering, color constraints and fill weight zero were preserved.

| Measurement | L2 | CLIP-convolution |
| --- | ---: | ---: |
| Complete planning | **146.12 s (2m26s)** | **163.00 s (2m43s)** |
| Pod startup | 142.01 s | 145.95 s |
| Successful-job client elapsed | **318.63 s (5m19s)** | **347.13 s (5m47s)** |
| Exported RGB mean squared pixel error | **0.01985** | 0.09143 |
| GPU region | EU-SE-1 | CA-MTL-1 |

Both successful Pods reported an NVIDIA A40, 9 vCPUs and $0.59/hour. These are
single runs on different hosts, not repeated hardware benchmarks. L2 took
10.35% less planning time and had 78.29% lower pixel MSE. Pixel errors compare
unmodified preview PNGs against the identical resized target in RGB `[0,1]`;
raw objective values were not compared. L2 better preserves the target's colors
and dark rocket silhouette, but both results remain coarse, with conspicuous
gaps and lost detail. CLIP retains some tower structure but changes the overall
brightness and colors substantially.

The comparison took **17m53s from the first launch until both downloads finished**,
including troubleshooting and two failed CLIP attempts in EU-SE-1:

1. `2018176e7f8549a8818c6eb721d9b0aa` failed at startup with `QueryError`.
   The original exception detail was not retained, so its cause is unknown.
   Before retrying, the only existing FRIDA Pod was matched to the L2 receipt.
2. `7b47c4cb5fa74c74b468b6a3d00c35c8` started but failed with
   `ClientConnectorError: No route to host` on the Runpod proxy. The SDK deleted
   its Pod, `zzmrtksyeewlg8`. Independent checks found no completed manifest or
   partial output files to recover.

The final CLIP attempt used CA-MTL-1. Its temporary client script retries only
read-only result polling for transient connection errors, timeouts and HTTP
502/503/504, at most six attempts. Submission, worker code and computation are
unchanged. Five isolated checks covered recovery and retry bounds; no retry
events occurred during the successful attempt. This workaround lives in the
reproduction script, not the production client or worker image.

All **46 files in each successful output** passed checksum verification. Both
saved plans contain 400 strokes and 12 palette colors. Local/downloaded model
weights and uploaded target hashes match; resized target PNGs are identical.
Successful Pods `q183bxnedxpvuj` and `3efo99dkpdnxcm`, plus failed Pod
`zzmrtksyeewlg8`, were independently confirmed absent. No FRIDA Pods remain.

Artifacts:

- L2: `outputs/runpod-starship/l2/70853f7ed9ee4b68aafc78c8676f735b/`
- CLIP: `outputs/runpod-starship/clip_conv_loss/6a449f6c3f9945fc94cb97b23ef244ba/`
- Each directory contains `preview.png`, `stroke_plan.json`, `palette.json`,
  the matching cache and intermediate previews.
- `outputs/runpod-starship/comparison.json` contains timings and pixel metrics;
  sibling receipts, runtime, verification and Pod metadata retain provenance.
- `outputs/runpod-starship/reproduce/` records the exact runner and verification
  script. The successful CLIP invocation used `FRIDA_RUN_DATACENTER=CA-MTL-1`
  and `FRIDA_RETRY_RESULT_POLL=1`; normal workflow defaults were not changed.

Target SHA-256:
`961a426957fcc6031cd60a9a8baf4b10e8950ad61ae87dbd7e441a7b6af73e7c`.
Worker `thaigrapecucumber/frida-worker:63555d6034a0eee3`, source fingerprint,
model/input version and weights are identical to the Michelangelo entry below.

## Claude target, same six-color setup — 2026-10-27

`targets/claude.jpg` completed on an NVIDIA A40 in CA-MTL-1 with the exact
model, materials, worker and planning settings used for Michelangelo below:
**L2, 400 strokes, 400 iterations, six colors**, batch size 64, seed zero,
21.6 × 13.9 cm canvas and 397 × 256 rendering. Only the target changed.
The existing worker matched local source, so no rebuild was necessary.

- Planning: **149.46 s (2m29s)**.
- Startup: **139.57 s (2m20s)**.
- Total client elapsed, including transfers, cleanup and validation:
  **328.68 s (5m29s)**.
- Worker input download: 6.88 s; output upload: 12.68 s.
- All **46 output files** passed checksum verification. The saved plan contains
  **400 strokes and six colors**, with the same retrained model as Michelangelo.
- Pod **t17jzpvyf5nmb9** was deleted; a separate API read confirmed no remaining
  FRIDA Pods. Reported rate was $0.59/hour. Credentials were discarded and
  temporary handoff helpers removed.

The actual simulation preview reduces the scene to broad, muted color regions;
fine harbor details, the small boats and the bright sun are not well reproduced.
The target is resized to the existing canvas proportions, as in prior runs.

Result directory:
`outputs/runpod-claude/28d3f06b57004c3e83cac36dd1920d8a/`.
It contains `preview.png`, `stroke_plan.json`, `palette.json`, the matching cache,
materials and intermediate previews. Sibling `-runtime.json`, `-verification.json`
and `-pod.json` retain timing and cleanup evidence. `outputs/runpod-claude/reproduce/run.py`
records the exact paid-job invocation.

Target SHA-256:
`7579e45242abd1934b57b207bc5f13872ee9b664aa17a876fcd96bf56c8e6416`.
Worker source and model/input version are identical to the Michelangelo entry.

## Michelangelo with the retrained model and six colors — 2026-10-27

The worker was rebuilt and published, then a normal planning job completed on
one NVIDIA A40 in CA-MTL-1 using `targets/michelangelo.jpg` and the new
`src/caches/so101_mvp` model. Requested settings were **L2, 400 strokes,
400 iterations, six colors**, batch size 64 and seed zero. Canvas dimensions
were **0.216 × 0.139 m**, matching the new library and `my_materials.json`;
the simulation renders at **397 × 256 pixels**.

| Measurement | Actual time |
| --- | ---: |
| Worker rebuild, import smoke test and publication | 28.06 s |
| Complete painting planning | **150.08 s (2m30s)** |
| Pod startup | 144.09 s (2m24s) |
| Total planning-job client elapsed, excluding image rebuild | **410.22 s (6m50s)** |

The worker downloaded inputs in 27.38 s and uploaded output in 21.36 s.
The total client time also includes local uploads/downloads, catalog lookup,
runtime orchestration, cleanup and validation. The new cache contains 120
recorded strokes, versus 40 in the earlier experiments, and is transferred as
an immutable bundle. Build time and time waiting for credential entry are not
part of the planning-job elapsed time.

All **46 output files** passed SHA-256 verification. The plan contains exactly
**400 strokes and six palette colors**. The target and model hashes were checked
against the uploaded manifests; downloaded weights match the new local model.
The model was reused for planning without retraining. Its recorded training
split is 96 training strokes and 24 validation strokes.

The simulation preview reproduces the broad background and hand shapes as flat
color regions, but much of the finger detail and shading is lost. This is the
actual six-color result, not an enhanced image. The palette was learned from
optimized stroke colors and clustered by the existing optimizer.

Pod **f4yp3fvxmyivdz** was deleted. A separate API read confirmed its absence and
no remaining FRIDA Pods. Reported rate was $0.59/hour; the 346.98 s task lifecycle
corresponds to about $0.057 of GPU time, an estimate excluding storage charges.
The credential process was closed and temporary handoff helpers removed.

Artifacts:

- `outputs/runpod-michelangelo/a525a9110e164f3d9e4b44db7143f863/preview.png`
- The same directory contains `stroke_plan.json`, `palette.json`, the matching
  new cache, materials, and previews every 50 iterations.
- Sibling `-runtime.json`, `-verification.json` and `-pod.json` files record
  exact timings, checksums, model identity, cleanup and allocated hardware.
- `outputs/runpod-michelangelo/reproduce/` contains the exact launch script and
  image build metadata. The launch script starts a paid A40 job when run with
  credentials in its environment; normal workflow defaults were not changed.

The published worker and `runpod.json` are now synchronized with local source,
including the robot-motion changes noted in the earlier L2 comparison:

```text
Image: thaigrapecucumber/frida-worker:63555d6034a0eee3
Manifest: sha256:a8a3790bd78318babe82ae441951f01b98f7c647d310188598283065b1a03e63
Source: 63555d6034a0eee35200b051240976d3081fb6e035ac21101fee22891cc57f6e
Model/input version: 744ad148569c8695e81f140d01ed089fc1235e2cdba293c98a3fe51e1a642495
Model weights SHA-256: cbc8bd5f54062a3f50ed9ef10bcd212d53a75f48caf967ea3c73f9ed384034ae
Target SHA-256: 7e4c77a7b454bec9c621074b58399d1496e885de3a912a101fbcc93739ba94d5
```

## L2 versus CLIP, 400 iterations — 2026-10-27

An L2 run completed on an NVIDIA A40 in CA-MTL-1 using the same immutable worker
image, SO-101 cache, target, seed, 400 strokes, 400 iterations, 12 palette colors,
render size and batch size as the CLIP run below. The **only planning-setting
difference was `objective="l2"`**. The two runs used separate Pods, each reporting
9 vCPUs and $0.59/hour. No optimizer code or normal workflow defaults were changed.

| Measurement | CLIP-convolution | L2 |
| --- | ---: | ---: |
| Complete planning | 164.12 s (2m44s) | **137.95 s (2m18s)** |
| Pod startup | 192.24 s | 162.98 s |
| Total client elapsed | 380.58 s (6m21s) | **326.32 s (5m26s)** |
| Exported RGB mean squared pixel error | 0.17109 | **0.03119** |
| Red target region with less than 5% paint opacity coverage | 18.733% | **0.0628%** |

L2 reduced measured planning time by **15.94%** and pixel error by **81.77%**.
Its red and blue blocks are substantially more filled, though edges remain
rough, white regions have unwanted marks, and the red is darker than the target.
These are single-run observations, not a general speed or quality guarantee.
Different startup durations also contribute to the end-to-end timing difference.

The original patchiness has concrete causes in this setup:

- `clip_conv_loss` compares intermediate CLIP features using cosine distance
  and augmentations; it does not directly minimize per-pixel RGB error. L2 is
  exactly `((painting - target)**2).mean()`. The controlled result supports L2
  as a better match for this target's flat colored regions and hard boundaries.
- The recorded settings have `fill_weight=0`, disabling the separate fill term.
  The optimizer always has 400 strokes; extra iterations adjust those strokes
  without adding new ones. Overlapping learned stroke footprints leave gaps.
- `BrushStroke.make_valid` clamps stroke lengths to 7–58 mm for this library.
  Saturated stroke colors are clamped to RGB components in `[0.02, 0.70]`;
  colors with all components at least 0.35 instead use an upper bound of 0.85.
  This limits vivid red reproduction independently of the selected objective.
  These constraints were preserved in both tests.

Pixel errors were computed on the exported preview PNGs against the identical
resized target PNG, with RGB values scaled to `[0,1]`. Raw CLIP and L2 objective
values were **not** compared. Paint coverage was reconstructed from each saved
plan on CPU as `1 - product(1 - stroke_alpha)`. The red-region mask is target
R > 0.75, G < 0.25, B < 0.25 (46,185 pixels). Whole-canvas coverage is not a
quality score because some target areas are intentionally white. CPU/CUDA
rounding can cause small differences in the reconstructed coverage.

All 30 L2 output files passed checksum verification. Both runs have identical
target renders, input/model versions, worker source fingerprints and settings
other than objective. The L2 plan has 400 strokes and 12 palette colors. Pod
**ts3nhg25cufxr4** was deleted; an independent API read confirmed no remaining
FRIDA Pods. Credentials were discarded and temporary handoff helpers removed.
At the reported rate, its 317.32 s task lifecycle is approximately $0.052 of GPU
time, excluding storage and subject to actual billing.

Artifacts:

- `outputs/runpod-l2-full400/82fc0fb9b20141e3b3536d0e1f4eb3a6/preview.png`
- The same directory contains `stroke_plan.json`, matching cache, palette and
  previews every 50 iterations.
- `outputs/runpod-l2-full400/comparison.json` records shared settings, pixel
  errors, reconstructed coverage, speed ratios and comparison caveats.
- Sibling job `-runtime.json`, `-verification.json` and `-pod.json` files retain
  timing, cleanup and allocation evidence. `reproduce/` contains the launch and
  read-only comparison scripts.

Concurrent local robot-motion edits changed `src/brush_stroke.py` while the L2
job ran. Both remote tests still used source `68489dacc4dda4bd…`; the rendering
methods used by the CPU coverage check are AST-identical to their prior version.
The physical-motion edits were left untouched. Local source now differs from
the configured worker, so it needs an image rebuild before another normal remote
launch. This does not invalidate the two completed tests.

## Full 400-iteration run — 2026-10-27

The normal worker completed **400 iterations and 400 strokes** on an NVIDIA A40
in CA-MTL-1, using `targets/images.jpeg`, the existing SO-101 MVP cache,
`clip_conv_loss`, batch size 64, 12 colors and a 256 × 330 canvas. This run used
the published renderer image below with no benchmark monkeypatches or optimizer
changes. CLIP was loaded normally inside the planning timer.

| Measurement | Actual time |
| --- | ---: |
| Complete worker planning, including CLIP load and final preview | **164.12 s (2m44s)** |
| Pod startup | 192.24 s (3m12s) |
| Total client elapsed time, including all transfers and cleanup | **380.58 s (6m21s)** |

The worker input download took 4.69 s and its output upload took 7.39 s; the
remaining elapsed time includes catalog lookup, local uploads/downloads,
runtime orchestration, termination and validation. At the reported $0.59/hour
rate, the 371.99 s task lifecycle corresponds to approximately **$0.061** of GPU
time, an estimate rather than an invoice and excluding storage charges.

All **30 output files** passed SHA-256 verification. The plan contains 400
strokes and a 12-color palette. The uploaded target's hash matches the local
image, and the worker source fingerprint matches the configured image. A
separate API read confirmed Pod **ocpi0hj43ntwnp** was deleted and no FRIDA Pods
remained. The temporary credential process was closed and its helpers removed.

The simulation preview captures the target's broad colored-block layout, but
coverage is patchy, with substantial white gaps and muted red. Completion of
400 iterations does not establish that the visual quality is sufficient for
physical painting. The robot was not operated.

Artifacts:

- `outputs/runpod-full400/cfc420ff60e740b6b3cc3c9a1b7659c7/preview.png`
- `outputs/runpod-full400/cfc420ff60e740b6b3cc3c9a1b7659c7/stroke_plan.json`
- The result directory also contains its matching cache, palette and intermediate
  previews every 50 iterations. `plan-401.png` is the final postprocessing log.
- Sibling `cfc420ff60e740b6b3cc3c9a1b7659c7-runtime.json` contains exact timings;
  `-verification.json` records checksum and deletion checks, and `-pod.json`
  records the reported GPU rate and hardware allocation.
- `outputs/runpod-full400/reproduce/run.py` preserves the exact launch settings.
  Running it with credentials in the environment starts another paid A40 job.

The ordinary `runpod_workflow.py` remains a 20-iteration test; this full run did
not change its default or any optimizer code.

## Renderer benchmark — 2026-10-27

The cached, batched renderer was tested live on **one NVIDIA A40 in CA-MTL-1**
using the exact `targets/images.jpeg`, existing SO-101 MVP model, 400 strokes,
20 iterations, `clip_conv_loss`, and a 256 × 330 canvas. All comparisons ran on
that same Pod with freshly seeded paintings and CLIP loaded before timing.

| Renderer | CPU threads | Complete 20-iteration plan | Median steady iteration |
| --- | ---: | ---: | ---: |
| Original | 48 (PyTorch default) | 222.54 s | 9.971 s |
| Original, reduced threads | 4 | 46.91 s | 2.103 s |
| Cached and batched, batch size 64 | 48 | **9.40 s** | **0.345 s** |

This is **23.66× faster complete planning**, or **28.92× faster median
iterations**, relative to the original default. Batching adds another 4.99×
planning speedup beyond reducing the original renderer's thread count.

The four-iteration thread sweep measured the following median steady iterations
(first iteration excluded). Differences of a few milliseconds are not strong
proof of an optimal thread count:

| CPU threads | Original | Batched |
| ---: | ---: | ---: |
| 1 | 2.041 s | 0.387 s |
| 2 | 2.056 s | 0.350 s |
| 4 | 2.039 s | 0.359 s |
| 8 | 2.102 s | 0.346 s |
| 48 | 9.971 s (20-iteration baseline) | 0.343 s |

Reducing CPU threads helps the original renderer dramatically; after batching,
the default is competitive. Normal remote planning therefore keeps
`cpu_threads=None` and defaults to `render_batch_size=64`. Both controls are
available as Python helper arguments; batch size zero selects the original path.

The short benchmark projected roughly 2–3 minutes for 400 iterations; the full
run above subsequently measured **164.12 s** for planning. Fresh Pod startup
remains separate: this comparison benchmark took **170.39 s to start**.
Planning times above exclude startup and S3 transfers. The benchmark uploaded
all intermediate runs (243 files), so its 63.27 s output upload is not a normal
single-plan transfer measurement. Its receipt covers the entire experiment,
and `result.json`'s 355.39 s planning time covers correctness checks plus all
12 comparison runs; use `benchmark/report.json` for individual plan timings.

### Correctness and preserved behavior

- The optimizer, color-ordering/shuffling functions, CLIP loss, brush-stroke
  implementation, and stroke-model implementation are byte-for-byte unchanged
  from before this renderer work. The original sequential compositing body is
  unchanged. Grid normalization, pivot, interpolation, padding and
  `align_corners=False` conventions are preserved.
- On the real 400-stroke CUDA painting, image RMS error was **0.000001997** and
  maximum pixel error **0.0001165** in the 0–1 range. The reference and batched
  CLIP losses were both **0.74267578125** with identical augmentations.
- Applying exactly the same CLIP image gradient to both renderers gave
  **0.0035% relative parameter-gradient error**. Full end-to-end CLIP gradients
  differed by 2.25% (cosine similarity 0.9997566), almost entirely explained by
  CLIP's sensitivity to the tiny changed pixels. Batching is numerically close,
  not bit-for-bit identical; optimization trajectories can differ. Final
  sampled losses after 20 iterations were 0.63721 (original default), 0.63574
  (original/four threads), and 0.63379 (batched), without claiming better quality.
- **98 local tests passed**, including pixel/gradient comparisons, clipping,
  overlapping strokes, ink, opacity, sorted/shuffled order, actual model batch
  sizes, remote transfers, and plan validation. Two existing recovery-mock cases
  were excluded because their synthetic namespace lacks `MotionPlanningError`;
  they also failed before renderer integration.
- All **243 downloaded files** passed SHA-256 verification. The main plan has
  **400 strokes**. Local and remote source fingerprints and target hashes match.
- Pod **u5y1bwfzbzgxck** was deleted. A separate API read confirmed its absence
  and **no remaining FRIDA Pods**. Actual reported rate was **$0.59/hour**.

Two preliminary diagnostic attempts stopped before timing and deleted their
Pods. The first exposed a benchmark RNG bug (first CLIP initialization consumed
random numbers); the second exceeded an overly coupled 2% full-CLIP gradient
check. The final harness adds a stricter isolated-renderer derivative check
(0.2% tolerance) and records full-CLIP gradients separately (5% / 0.9995 cosine
guard). Thresholds, results and diagnostic history are saved in the provenance.
No optimizer change was made to make these comparisons pass.

Artifacts:

- `outputs/renderer-benchmark/5551e43b08f243d1a12827dafc6cc365/benchmark/report.json`
  contains every iteration timing, sampled loss, thread setting and CUDA check.
- The sibling `.json` receipt and `-verification.json` record lifecycle and
  independent deletion verification. The result directory contains the main
  `stroke_plan.json`, `preview.png`, matching cache and intermediate plans.
- `outputs/renderer-benchmark/reproduce/` preserves the temporary benchmark
  harness, provenance and source-preservation checks, without credentials.

The published and locally configured worker is now:

```text
thaigrapecucumber/frida-worker:68489dacc4dda4bd
Manifest: sha256:308e8aeee1ccbe028614ed25740527aafdfe6993a4d2da4f0c8bb1dfc9fcd979
Source: 68489dacc4dda4bda59e26915194403764b789dede165042dd245f708ac7226f
```

Run `.venv-runpod/bin/python runpod_workflow.py` as before. Its 20-iteration test
now uses the batched renderer automatically. The same storage volume, credentials,
automatic GPU selection and $2/hour catalog limit remain configured.

## Initial live validation — 2026-10-26

A live CUDA painting run has now succeeded for the exact `targets/images.jpeg`
with the existing SO-101 cache. The remote input's SHA-256 matches the local file.

- GPU: NVIDIA A40, EU-SE-1, actual reported Pod rate $0.59/hour.
- Objective: clip_conv_loss, 400 strokes, 20 iterations, render height 256.
- Host CUDA capability: 13.2; worker PyTorch 2.11.0+cu128; recorded device: cuda.
- Planning: 222.44 seconds; startup: 183.42 seconds; total client time: 427.57 seconds.
- Remote input download: 2.80 seconds; output upload: 3.63 seconds.
- All 22 output-file hashes passed verification, including a 400-stroke plan.
- Pod aa1k5pz73c8tiw was deleted; a separate Runpod API read confirmed its absence
  and that no FRIDA Pods remained. The temporary credential process was closed.

Outputs are in `outputs/runpod/dd3aedacae574c8a8fff9d860226b8ec/`. The sibling
`.json` receipt records the task lifecycle; `-verification.json` records the
independent target-hash and deletion checks. This is a 20-iteration validation
run, not a fully converged painting or evidence of near-instant throughput.

The live diagnosis found that all currently available hosts advertised CUDA 13.2,
while the pinned SDK's built-in list ended at 13.0. The previous exact-version
catalog filter returned zero available types; querying a 12.8 minimum returned
22 available types. Discovery now uses that minimum and adds compatible versions
from the live catalog to the SDK's deployment filter in memory.

- **51 tests passed** across `src/tests/test_remote.py` and
  `src/tests/test_runpod_workflow.py` (10.75 seconds). These include real CPU
  planning/training, all three operations through S3 bundle transfers, model
  reuse, checksums, cached calibration consistency, and cleanup on failure.
- **Actual pinned SDK payload checked:** Iceland storage with a GPU in EU-SE-1,
  no network-volume mount, exactly one GPU, and credentials only in runtime env.
  Missing live-catalog regions are registered in the SDK enum in memory.
- **Capacity fallback checked:** an explicit deployment capacity rejection tries
  the next GPU/region. Timeouts, authentication failures, and failures after spawn
  never launch another job. The catalog filter considers every region, at least
  24 GB VRAM, SDK-supported NVIDIA types, live compatible CUDA versions, and the
  user's $2/hour catalog-price ceiling. Prices remain estimates.
- **One-session workflow checked:** keys are prompted once. Availability failure
  can be retried using the same process and credentials. Target and cache stay
  `targets/images.jpeg` and `src/caches/so101_mvp`.
- **Published container smoke test passed with outbound networking disabled.**
  The real SO-101 cached model and target were mounted read-only into the image.
  Real Boto3 clients made signed HTTP requests to a loopback mock S3 server:
  46 GETs and 43 PUTs. The worker downloaded inputs to temporary disk, performed
  four-stroke/one-iteration L2 planning on CPU, uploaded results, and the client
  downloaded and validated the resulting plan after temporary-worker cleanup.
  This earlier CPU smoke test complemented the successful live CUDA/S3 run above.
- **Runtime imports, source fingerprint, client dependency check and syntax/
  whitespace checks passed.** Runpod's image-pull logs verified the published manifest;
  runpod.json points to the matching version.

Published image:

```text
thaigrapecucumber/frida-worker:0f297a84b0d538e9
Manifest: sha256:83c8e45039e4975feed07fd12542434cb893563c48ce9e63e6f55edc18f14bbf
Image config: sha256:54cdd17764ff2268c5b5a6073006f067623062c35b31ced32e119ca49a35ecfc
Source: 0f297a84b0d538e93970294e6106be5dd3cee5cd5990070a6256bbf89a173550
Linux/amd64; Python 3.12; PyTorch 2.11.0; CUDA 12.8; CLIP ViT-B/32 weights included
```

The updated local configuration retains volume `u8lwd7nc27` and its S3 region
`EUR-IS-1`. It sets `gpu: auto`, `compute_datacenter: null`, a $2/hour catalog
limit, and up to 12 placement attempts. No new volume or account credentials are
needed. The script defaults to the paid 400-stroke/20-iteration CLIP planning test.

Earlier validation also exercised actual SO-101 CLIP-convolution forward/backward
on the Mac CPU. Existing unrelated stroke-plan/reload tests had 36 passes and
three pre-existing failures because their mocked execution environment did not
define MotionPlanningError; they were not changed for this update.

CUDA brush-model training remains unverified live; its transfer/training path
passes CPU tests. The completed live run verifies cross-region S3, CUDA CLIP
planning, result retrieval, and Pod termination. See [the setup guide](runpod.md).

## Black-bear foreground planning — 2026-10-28

Implemented opt-in `foreground_mode=True` for black silhouettes on a white canvas. This places complete learned brush footprints inside the foreground, relocates/removes unhelpful strokes every 50 iterations, and prunes again after optimization. The requested stroke count is a budget. Full brush widths and texture are checked; no pixels are masked out of the preview. Final retained strokes allow at most 2% of predicted paint mass outside the silhouette.

Both objectives add a full-canvas background MSE penalty (weight 4); CLIP also adds full-canvas reconstruction MSE (weight 1). All six jobs used A40, 400 iterations, seed 0, 397 × 256 rendering, batch 64, CPU threads unset, the same cached SO101 model and materials, and the solid-black swatch through `use_colors_from`. Renderer, physical limits, compositing, color order and RMSprop schedule are unchanged.

Worker: `thaigrapecucumber/frida-worker:0a719b8439971127`. Source: `0a719b843997112786b2469d550945ffba6a018be15f126155ab065d087ce2ae`. Baseline: `outputs/runpod-black-bear-locked`. Results and reproduction scripts: `outputs/runpod-black-bear-foreground`.

| Objective | Budget | Retained | Planning seconds | Prior IoU | New IoU | Background marked |
|---|---:|---:|---:|---:|---:|---:|
| L2 + background | 100 | 93 | 42.77 | 73.63% | 96.60% | 0.050% |
| L2 + background | 196 | 153 | 61.91 | 65.83% | 98.25% | 0.038% |
| L2 + background | 400 | 184 | 82.64 | 58.39% | 98.54% | 0.029% |
| CLIP + L2 + background | 100 | 92 | 57.83 | 34.31% | 93.21% | 0.089% |
| CLIP + L2 + background | 196 | 145 | 85.16 | 37.18% | 96.95% | 0.069% |
| CLIP + L2 + background | 400 | 184 | 100.73 | 31.21% | 98.15% | 0.054% |

Planning time includes foreground initialization, refinement, optimization and export. Successful client runtimes were 136.6–302.6 seconds, including startup/transfers/cleanup. One capacity rejection and one unavailable-region check started no GPU. A later container stalled before becoming ready, was cancelled with independent deletion verification, and the last job was successfully retried in Canada. Timings are single observations on separate A40 hosts.

Validation: 143 focused tests passed; a final 9-test silhouette check also passed. Disabled-mode regression matched the previous learned-palette, multi-color, ink and fixed-black behavior. Every output manifest, target/model identity, black palette, iteration log, full stroke footprint and CPU reload was checked. All six plans passed the actual `paint.py --simulate --load_plan` entry point with unchanged checkpoints. Runpod independently reported no remaining Pods, including the cancelled startup. The memory-only credential process was closed. No physical robot execution was performed.
