from options import Options
from camera.innomaker import WebCam
import matplotlib.pyplot as plt
import os
import time
import cv2

def test_capture_image(camera: WebCam):
    path, img = camera.capture_image()
        
    if path is None:
        raise ValueError("Path is None!")
    else:
        print(path)
        
def test_get_canvas(camera: WebCam):
    canvas = camera.get_canvas()
    
    capture_dir = 'captures'
    os.makedirs(capture_dir, exist_ok=True)
    file_name = os.path.join(capture_dir, str(time.time()) + '.jpg')
    success = cv2.imwrite(file_name, canvas)
    
    if not success:
        raise FileExistsError("Failed to save image to ", file_name)
    

if __name__ == '__main__':
    opt = Options()
    opt.gather_options()
    opt.CANVAS_WIDTH_M = 0.279
    opt.CANVAS_HEIGHT_M = 0.216 
    camera = WebCam(opt)
    
    # test_capture_image(camera)
    test_get_canvas(camera)
    