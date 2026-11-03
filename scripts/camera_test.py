
if __package__:
    from ._bootstrap import ROOT, SOURCE_DIR
else:
    from _bootstrap import ROOT, SOURCE_DIR

from camera.dslr_gphoto import *

cam = camera_init()

path, img = capture_image(cam)
path, img = capture_image(cam)
path, img = capture_image(cam)

import matplotlib.pyplot as plt
plt.imshow(img)
#plt.show()
