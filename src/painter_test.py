import sys
import datetime
import numpy as np

import torch
from paint_utils3 import canvas_to_global_coordinates, get_colors, nearest_color, random_init_painting, save_colors, show_img

from painter import Painter
from options import Options
# from paint_planner import paint_planner_new

from my_tensorboard import TensorBoard
from painting_optimization import load_objectives_data, optimize_painting

# from tests.paint_hello_world import paint_hello_world

if __name__ == '__main__':
    opt = Options()
    
    opt.robot = "so101"
    opt.use_cache = True
    opt.cache_dir = 'caches/sharpie_short_strokes'
    opt.dont_retrain_stroke_model = True
    opt.simulate = False
    opt.materials_json = '../my_materials.json'
    opt.gather_options()
    
    date_and_time = datetime.datetime.now()
    run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
    opt.writer = TensorBoard('{}/{}'.format(opt.tensorboard_dir, run_name))
    opt.writer.add_text('args', str(sys.argv), 0)
    painter = Painter(opt)
    opt.HOVER_FACTOR = 0.05

    try:
        # Test to neutral
        # painter.to_neutral()
        
        # painter.rub_brush_on_rag()
        painter.to_neutral()
        # painter.dip_brush_in_water()
        # painter.get_paint(1)
        # paint_hello_world(painter, air_test=False)
        # # painter.clean_paint_brush()
        painter.hover_above_positions()
        # painter.to_neutral()
        # painter.clean_paint_brush()
        painter.to_neutral()
    finally:
        painter.robot.good_night_robot()
    
    # painter.clean_paint_brush()
    # painter.get_paint(color_ind)