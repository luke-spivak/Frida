"""Press Run for the 20-iteration painting test on an available GPU in any region.

Credentials are prompted once and kept only in this process. Use --availability
for a read-only catalog check, or ACTION='check' for local setup diagnostics.
GPU selection and the $2/hour catalog limit are configured in src/remote/runpod.json.
"""

if __package__:
    from ._bootstrap import ROOT, SOURCE_DIR
else:
    from _bootstrap import ROOT, SOURCE_DIR


import argparse
from pathlib import Path

from remote import plan_remotely, train_brush_remotely, train_and_plan_remotely
from remote.availability import AvailabilityError, require_capacity
from remote.config import RunpodConfig
from remote.setup import doctor, prompt_credentials

ACTION = "plan"
CACHE = ROOT / "src/caches/so101_mvp"
MATERIALS = ROOT / "my_materials.json"
TARGET = ROOT / "targets/images.jpeg"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--availability", action="store_true", help="List capacity without starting a GPU")
    palette_group = parser.add_mutually_exclusive_group()
    palette_group.add_argument("--use_colors_from", type=Path, help="Extract a fixed palette from this image")
    palette_group.add_argument("--palette", nargs="+", metavar="#RRGGBB", help="Use these fixed paint colors")
    parser.add_argument("--n_colors", type=int, help="Palette size; explicit colors infer their count")
    args = parser.parse_args()
    palette_options = {name: getattr(args, name) for name in ("use_colors_from", "palette", "n_colors")
                       if getattr(args, name) is not None}
    if ACTION == "train" and palette_options and not args.availability:
        parser.error("Palette options apply to planning, not brush-model training")
    if ACTION == "check" and not args.availability:
        return 1 if doctor() else 0
    config = RunpodConfig.load()
    readonly = args.availability or ACTION == "availability"
    prompt_credentials(api_only=readonly)
    while True:
        try:
            if readonly:
                for candidate in require_capacity(config):
                    print(f"{candidate['gpu']} | {candidate['datacenter']} | "
                          f"{candidate['stock']} | ~${candidate['catalog_price']:.2f}/hour")
                print("Read-only check complete. No GPU started.")
                return 0
            if ACTION in ("plan", "choose_gpu"):
                result = plan_remotely(target=TARGET, cache_dir=CACHE, materials_json=MATERIALS,
                                       config=config, objective="clip_conv_loss", num_strokes=400, iterations=20,
                                       **palette_options)
            elif ACTION == "train":
                model = train_brush_remotely(cache_dir=CACHE, materials_json=MATERIALS, config=config)
                print("New model version:", model.version)
                print("New cache:", model.cache_dir)
                return 0
            elif ACTION == "train_and_plan":
                result = train_and_plan_remotely(target=TARGET, cache_dir=CACHE, materials_json=MATERIALS,
                                                config=config, objective="clip_conv_loss",
                                                num_strokes=400, iterations=400, **palette_options)
            else:
                raise ValueError(f"Unknown ACTION: {ACTION}")
            print("Preview:", result.preview_path)
            print("Plan:", result.plan_path)
            print("Matching cache:", result.model.cache_dir)
            return 0
        except AvailabilityError as exc:
            print(exc)
            if readonly or input("Enter to refresh capacity with the same keys, or q to quit: ").strip():
                return 1
        except KeyboardInterrupt:
            return 130
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}")
            print("Stopped without retrying. If startup or cleanup was unconfirmed, check the Runpod console.")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
