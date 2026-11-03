"""Measure brush deterioration across reloads, then test recovery after washing.

Default: wash/wipe, six loads of five identical strokes with no intervening
washing, then wash/wipe and paint one recovery load. Preview unless --execute.
"""

if __package__:
    from ._bootstrap import ROOT, SOURCE_DIR
else:
    from _bootstrap import ROOT, SOURCE_DIR


import argparse
from datetime import datetime
from pathlib import Path
import time
from uuid import uuid4

from paint_loading import (STROKES_PER_LOAD, STARTUP_PROMPT, StartupPreflightError, add_common_arguments, base_report,
                           execute_test, load_setup, make_layout, parse_common_arguments,
                           repeatable_loading, save_preview, save_report)


def arguments(argv=None):
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__), "paint-washing")
    parser.add_argument("--reload-every", type=int, default=5,
                        help="Strokes per load; use the interval established by the loading test (2-10)")
    parser.add_argument("--batches", type=int, default=6,
                        help="Loads before the recovery wash (2-20); default 6 x 5 = 30 strokes")
    args = parse_common_arguments(parser, argv)
    if not 2 <= args.reload_every <= STROKES_PER_LOAD:
        parser.error("--reload-every must be between 2 and 10")
    if not 2 <= args.batches <= 20:
        parser.error("--batches must be between 2 and 20")
    return args


def make_protocol(args, layout):
    """Keep whole loads on a sheet; the recovery always gets a fresh sheet.

    Slots repeat across sheets, allowing comparisons at the same location and
    same position within a load. Recovery uses the initial baseline's slots.
    """
    per_sheet = len(layout) // args.reload_every
    sheets, number = [], 0
    for index in range(args.batches + 1):
        recovery = index == args.batches
        if recovery or index % per_sheet == 0:
            sheets.append({"sheet": len(sheets) + 1, "status": "pending", "batches": []})
        sheet = sheets[-1]
        offset = 0 if recovery else (index % per_sheet) * args.reload_every
        batch = {"batch": index + 1, "phase": "recovery" if recovery else ("baseline" if index == 0 else "unwashed"),
                 "wash_before": index == 0 or recovery, "status": "pending", "completed": 0, "strokes": []}
        for within, mark in enumerate(layout[offset:offset + args.reload_every], 1):
            number += 1
            batch["strokes"].append({**mark, "stroke": number, "sheet": sheet["sheet"],
                                     "batch": batch["batch"], "phase": batch["phase"],
                                     "stroke_in_load": within,
                                     "stroke_since_wash": within if recovery else number})
        sheet["batches"].append(batch)
    # Second-stroke comparisons are listed explicitly; grouping by slot avoids
    # treating positional differences as evidence that the brush needs washing.
    comparisons = {}
    for sheet in sheets:
        for batch in sheet["batches"]:
            mark = batch["strokes"][1]
            comparisons.setdefault(str(mark["slot"]), []).append({
                key: mark[key] for key in ("stroke", "sheet", "batch", "phase", "stroke_since_wash")})
    return {"sheets": sheets, "compare_second_strokes_by_slot": comparisons,
            "completed_strokes": 0, "strokes_since_wash": 0, "in_progress": None}


def run_batches(args, painter, stroke, report, directory, recording, snapshot,
                execute_stroke, prompt=input, clock=time.monotonic):
    """Keep wash age across sheets; abort on any partial/failed motion."""
    report_path = directory / "test.json"
    last_wash_end = None
    for sheet in report["sheets"]:
        painter.to_neutral()
        recovery = sheet["batches"][0]["phase"] == "recovery"
        recording.event("paper_wait_start", sheet=sheet["sheet"],
                        strokes_since_wash=report["strokes_since_wash"])
        started = clock()
        instruction = ("The robot will wash and load for the recovery check." if recovery else
                       "Do not manually wash, wipe, or add paint to the brush between batches.")
        prompt(f"\nSheet {sheet['sheet']}/{len(report['sheets'])}: place FRESH paper in the same position. "
               f"Black paint in slot 0. {instruction} Keep paper changes brief and consistent. "
               "Press Enter when hands are clear (Ctrl-C cancels): ")
        sheet["paper_wait_seconds"] = clock() - started
        recording.event("paper_ready", sheet=sheet["sheet"], wait_seconds=sheet["paper_wait_seconds"])
        snapshot(f"sheet-{sheet['sheet']:02d}-before")
        sheet["status"] = "running"
        save_report(report_path, report)
        for batch in sheet["batches"]:
            batch["status"] = "running"
            save_report(report_path, report)
            if batch["wash_before"]:
                recording.event("clean_start", batch=batch["batch"], phase=batch["phase"],
                                passes=args.wash_passes, strokes_since_wash=report["strokes_since_wash"])
                with repeatable_loading(args.seed):
                    for _ in range(args.wash_passes):
                        painter.clean_paint_brush()
                report["strokes_since_wash"] = 0
                last_wash_end = clock()
                recording.event("clean_end", batch=batch["batch"], phase=batch["phase"])
                save_report(report_path, report)
            # Seed the dabs independently of washing: baseline, unwashed and
            # recovery loads must all use exactly the same commanded offsets.
            recording.event("reload_start", batch=batch["batch"], phase=batch["phase"], palette_slot=0,
                            strokes_since_wash=report["strokes_since_wash"])
            with repeatable_loading(args.seed):
                painter.get_paint(0)
            load_end = clock()
            recording.event("reload_end", batch=batch["batch"], phase=batch["phase"])
            for mark in batch["strokes"]:
                report["in_progress"] = mark["stroke"]
                save_report(report_path, report)
                print(f"{batch['phase']}: stroke {mark['stroke']}, "
                      f"{mark['stroke_in_load']}/{args.reload_every} in load, "
                      f"{mark['stroke_since_wash']} since wash", flush=True)
                recording.event("stroke_start", **mark, seconds_since_wash=clock() - last_wash_end,
                                seconds_since_load=clock() - load_end)
                result = execute_stroke(stroke, painter, mark["x"], mark["y"], 0.)
                if result is False:
                    raise RuntimeError("Stroke was not executed; washing test is incomplete")
                batch["completed"] += 1
                report["completed_strokes"] += 1
                report["strokes_since_wash"] += 1
                report["in_progress"] = None
                save_report(report_path, report)
                recording.event("stroke_end", **mark, seconds_since_wash=clock() - last_wash_end,
                                seconds_since_load=clock() - load_end)
            batch["status"] = "completed"
            save_report(report_path, report)
            recording.event("batch_end", batch=batch["batch"], phase=batch["phase"],
                            strokes_since_wash=report["strokes_since_wash"])
        painter.to_neutral()
        snapshot(f"sheet-{sheet['sheet']:02d}-after")
        sheet["status"] = "completed"
        save_report(report_path, report)


def main(argv=None):
    args = arguments(argv)
    opt, params, grid, homography, xy_correction = load_setup(args)
    layout = make_layout(opt, grid, args.length_mm / 1000, xy_correction)
    report = base_report(args, params, xy_correction, Path(__file__))
    report.update(make_protocol(args, layout))
    report.update({
        "experiment": "washing",
        "sequence_protocol": "Initial wash/wipe; fixed paint reloads with no intervening washes; final wash/wipe and recovery load",
        "interpretation": "Compare the second stroke of each load, grouped by slot. Check baseline vs recovery. "
                          "Paper-change pauses and paint drying can affect results; no automatic wash interval is inferred.",
    })
    name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    directory = args.output_dir / name
    directory.mkdir(parents=True, exist_ok=False)
    for sheet in report["sheets"]:
        marks = [mark for batch in sheet["batches"] for mark in batch["strokes"]]
        save_preview(directory / f"sheet-{sheet['sheet']:02d}-layout.svg", opt, marks, args.length_mm / 1000)
    save_report(directory / "test.json", report)
    print(f"Initial wash: {args.wash_passes} water/rag cycles. Then {args.batches * args.reload_every} strokes, "
          f"reloading every {args.reload_every}, WITHOUT washing.")
    print(f"Recovery: another {args.wash_passes} water/rag cycles, reload, {args.reload_every} more strokes.")
    print(f"Total: {(args.batches + 1) * args.reload_every} strokes on {len(report['sheets'])} fresh sheets.")
    print(f"Straight {args.length_mm:g} mm strokes; compression: middle {report['compression_mm']['middle']:.2f} mm, "
          f"endpoints {report['compression_mm']['endpoints']:.2f} mm (commanded, not force).")
    print(report["xy_frame"] + ". No trained model required.")
    print("Use a reload interval validated by the loading test; change it with --reload-every.")
    for slot, marks in report["compare_second_strokes_by_slot"].items():
        print(f"Compare second strokes at slot {slot}: " + ", ".join(str(m["stroke"]) for m in marks))
    print(f"Layouts/settings: {directory}")
    if not args.execute:
        print("Preview only: no robot/camera connection or IK check. Add --execute to paint and record.")
        return 0
    input(STARTUP_PROMPT)
    execute_test(args, opt, params, grid, homography, xy_correction, directory, report,
                 sequence_runner=run_batches)
    print(f"Completed. Photos, layouts and test.json: {directory}; video/timing: {directory / 'recording'}")
    print("Compare identical load positions and canvas slots, then recheck after drying. "
          "If a long paper change affected a batch, use the recorded timings when interpreting it.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StartupPreflightError as exc:
        print(f"\nStopped: {exc}")
        raise SystemExit(2)
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. The partial test cannot resume: start a new test with fresh paper and the initial wash.")
        raise SystemExit(130)
