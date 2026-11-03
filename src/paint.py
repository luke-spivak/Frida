#! /usr/bin/env python
##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

import sys
import cv2
import datetime
from pathlib import Path
import numpy as np

import torch
from paint_utils3 import canvas_to_global_coordinates, get_colors, nearest_color, random_init_painting, save_colors, show_img, format_img

from painter import Painter
from options import Options
from robot import MotionPlanningError
# from paint_planner import paint_planner_new

from my_tensorboard import TensorBoard
from stroke_plan import DEFAULT_FILENAME, StrokePlan
from paint_recording import record_painting

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    opt = Options()
    opt.gather_options()

    load_path = opt.load_plan or (Path(opt.cache_dir) / DEFAULT_FILENAME if opt.dont_plan else None)
    plan_path = Path(opt.save_plan or load_path or Path(opt.cache_dir) / DEFAULT_FILENAME)
    loaded_plan = StrokePlan.load(load_path) if load_path else None
    if loaded_plan is not None:
        loaded_plan.check_setup(opt)  # Validate before Painter connects or moves.
        if loaded_plan.remaining == 0:
            print('Saved plan is complete; there are no strokes left to paint.')
            sys.exit(0)
        opt.dont_retrain_stroke_model = True
        opt.render_height = loaded_plan.data['render_height']
        print(f'Loaded {load_path}: {loaded_plan.completed} completed, '
              f'{loaded_plan.remaining} remaining. Optimization will be skipped.')
        if loaded_plan.data['in_progress']:
            print(f'The previous run stopped during stroke {loaded_plan.completed + 1}; '
                  'it will be retried. Inspect the canvas before closing the ready window.')
    elif opt.num_adaptations < 1 or opt.num_strokes < 1:
        raise ValueError('num_adaptations and num_strokes must be positive')

    date_and_time = datetime.datetime.now()
    run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
    opt.writer = TensorBoard('{}/{}'.format(opt.tensorboard_dir, run_name))
    opt.writer.add_text('args', str(sys.argv), 0)

    painter = Painter(opt)
    opt = painter.opt # This necessary?

    with record_painting(painter, opt, plan_path) as recording:
        if not opt.simulate and loaded_plan is None:
            try:
                input('Make sure blank canvas is exposed. Press enter when you are ready for the paint planning to start. Use tensorboard to see which colors to paint.')
            except SyntaxError:
                pass

        # paint_planner_new(painter)

        painter.to_neutral()

        w_render = int(opt.render_height * (opt.CANVAS_WIDTH_M/opt.CANVAS_HEIGHT_M))
        h_render = int(opt.render_height)
        opt.w_render, opt.h_render = w_render, h_render

        consecutive_paints = 0
        consecutive_strokes_no_clean = 0
        curr_color = -1

        color_palette = None
        if loaded_plan is None and opt.use_colors_from is not None:
            color_palette = get_colors(cv2.resize(cv2.imread(opt.use_colors_from)[:,:,::-1], (256, 256)),
                    n_colors=opt.n_colors).to(device)
            opt.writer.add_image('paint_colors/using_colors_from_input', save_colors(color_palette), 0)

        current_canvas = painter.camera.get_canvas_tensor(h=h_render,w=w_render).to(device) / 255.

        if loaded_plan is None:
            from painting_optimization import load_objectives_data, optimize_painting
            load_objectives_data(opt)
            painting = random_init_painting(opt, current_canvas, opt.num_strokes, ink=opt.ink, device=device)
            painting.to(device)
            painting, color_palette = optimize_painting(opt, painting,
                        optim_iter=opt.init_optim_iter, color_palette=color_palette)
            checkpoint = StrokePlan.capture(painting, color_palette, opt, nearest_color)
        else:
            checkpoint = loaded_plan
            painting, color_palette = checkpoint.restore(opt, current_canvas, device)
            with torch.no_grad():
                opt.writer.add_image('images/plan', format_img(
                    painting(h_render, w_render, use_alpha=False)), 0)
        checkpoint.save(plan_path)
        print(f'Stroke plan checkpoint: {plan_path.resolve()}')

        if color_palette is not None:
            opt.writer.add_image('paint_colors/palette', save_colors(color_palette), 0)
            colors = (color_palette.detach().cpu().numpy() * 255).astype(np.uint8)
            np.save(Path(opt.cache_dir) / 'colors_updated.npy', colors)
            for index, color in enumerate(colors):
                print(f'Palette slot {index}: RGB {color.tolist()}')

        if not painter.opt.simulate:
            show_img(painter.camera.get_canvas()/255.,
                     title=("Saved plan loaded. Keep the existing canvas. " if loaded_plan is not None
                            else "Initial plan complete. ") +
                           "Ensure paints match the saved palette slots, then close this to start painting.")

        if opt.simulate:
            print('Simulation: plan saved; no strokes marked completed.')
            painter.robot.good_night_robot()
            sys.exit(0)

        strokes_per_adaptation = (len(painting) if loaded_plan is not None
                                  else max(1, int(len(painting) / opt.num_adaptations)))
        # for adaptation_it in range(opt.num_adaptations):
        while len(painting) > 0:
            ################################
            ### Execute some of the plan ###
            ################################
            for stroke_ind in range(min(len(painting),strokes_per_adaptation)):
                stroke = painting.brush_strokes[0]

                # Clean paint brush and/or get more paint
                if not painter.opt.ink:
                    color_ind = checkpoint.next_palette_index
                    new_paint_color = color_ind != curr_color
                    if new_paint_color or consecutive_strokes_no_clean >= 12:
                        recording.event("clean_start", stroke=checkpoint.completed + 1,
                                        palette_index=color_ind,
                                        completed_since_clean=consecutive_strokes_no_clean)
                        painter.clean_paint_brush()
                        recording.event("clean_end", stroke=checkpoint.completed + 1)
                        consecutive_strokes_no_clean = 0
                        curr_color = color_ind
                        new_paint_color = True
                    if consecutive_paints >= opt.how_often_to_get_paint or new_paint_color:
                        recording.event("reload_start", stroke=checkpoint.completed + 1,
                                        palette_index=color_ind,
                                        completed_since_reload=consecutive_paints)
                        painter.get_paint(color_ind)
                        recording.event("reload_end", stroke=checkpoint.completed + 1,
                                        palette_index=color_ind)
                        consecutive_paints = 0

                # Convert the canvas proportion coordinates to meters from robot
                x, y = stroke.transformation.xt.item()*0.5+0.5, stroke.transformation.yt.item()*0.5+0.5
                y = 1-y
                x, y = min(max(x,0.),1.), min(max(y,0.),1.) #safety
                x_glob, y_glob,_ = canvas_to_global_coordinates(x,y,None,painter.opt)

                # Runnit
                checkpoint.begin_stroke(plan_path)
                recording.event("stroke_start", stroke=checkpoint.completed + 1)
                try:
                    stroke.execute(painter, x_glob, y_glob, stroke.transformation.a.item())
                except MotionPlanningError as exc:
                    recording.event("stroke_failed", stroke=checkpoint.completed + 1, error=str(exc))
                    if not opt.skip_failed_strokes:
                        raise
                    stroke_number = checkpoint.completed + 1
                    print(
                        f"Planning failed for stroke {stroke_number}: {exc}", flush=True,
                    )
                    print(
                        "Attempting to return to neutral before skipping it; "
                        "the stroke may be partially painted.", flush=True,
                    )
                    try:
                        painter.to_neutral()
                    except Exception as recovery_error:
                        print(
                            "Could not recover to neutral; leaving the stroke pending "
                            f"and stopping: {recovery_error}", flush=True,
                        )
                        raise
                    checkpoint.skip_stroke(plan_path, str(exc))
                    recording.event("stroke_skipped", stroke=stroke_number, error=str(exc))
                    painting.pop()
                    print(
                        f"Skipped stroke {stroke_number}; checkpoint advanced.", flush=True,
                    )
                    continue
                except BaseException as exc:
                    recording.event("stroke_failed", stroke=checkpoint.completed + 1,
                                    error=f"{type(exc).__name__}: {exc}")
                    raise
                if not painter.opt.ink:
                    # Only completed strokes count toward reload and cleaning intervals.
                    consecutive_paints += 1
                    consecutive_strokes_no_clean += 1
                checkpoint.finish_stroke(plan_path)
                recording.event("stroke_end", stroke=checkpoint.completed)
                painting.pop()

            #######################
            ### Update the plan ###
            #######################
            painter.to_neutral()
            current_canvas = painter.camera.get_canvas_tensor(h=h_render,w=w_render).to(device) / 255.
            painting.background_img = current_canvas
            if loaded_plan is None and len(painting) > 0:
                painting, _ = optimize_painting(opt, painting,
                            optim_iter=opt.optim_iter, color_palette=color_palette)
                checkpoint.update_remaining(painting, nearest_color)
                checkpoint.save(plan_path)



        # to_video(plans, fn=os.path.join(opt.plan_gif_dir,'sim_canvases{}.mp4'.format(str(time.time()))))
        # with torch.no_grad():
        #     save_image(painting(h*4,w*4, use_alpha=False), os.path.join(opt.plan_gif_dir, 'init_painting_plan{}.png'.format(str(time.time()))))


        painter.robot.good_night_robot()


