#! /usr/bin/env python3
##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

import numpy as np
import torch
from torchvision.transforms import Resize


class SimulatedWebCam():
    def __init__(self, opt):
        self.opt = opt
        w_h_ratio = float(opt.CANVAS_WIDTH_M) / opt.CANVAS_HEIGHT_M
        h = 1024
        self.canvas = np.ones((h,int(h * w_h_ratio),3), dtype=np.float32) * 255.
    def get_canvas(self):
        return self.canvas
    def get_canvas_tensor(self, h=None, w=None):
        canvas = self.get_canvas()
        canvas = torch.from_numpy(canvas).permute(2,0,1).unsqueeze(0)
        if h is not None and w is not None:
            canvas = Resize((h,w), antialias=True)(canvas)
        canvas = torch.cat([canvas, torch.ones(1,1,h,w)], dim=1)
        return canvas
    def calibrate_canvas(self, use_cache=False):
        pass
