import os
import sys
import torch
import time

import cv2
import numpy as np

filePath = os.path.realpath(__file__)
project_path = os.path.dirname(os.path.dirname(os.path.dirname(filePath)))
sys.path.append(os.path.join(project_path, 'src'))
from xttmp.util.iostream import FrameIterator, FrameVisualizer
from xttmp.api import (instancing_model, inference) # type: ignore
from xttmp.util.compute_module import PostProcessing, AreaNMS, matrix_to_sparse_list # type: ignore


# DEVICE = 'cpu' # 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

import logging
logging.basicConfig(level=logging.INFO)

def test():
    # model
    model = instancing_model('vSTMD_F', device=DEVICE) 
    # input
    # frame_reader = FrameIterator(os.path.join(project_path, 'example-data', 'RIST_GX010290_orignal_240Hz.mp4'), is_video=True)
    # frame_reader = FrameIterator(os.path.join('D:/', 'STMD_Dataset', 'vSTMD_Panorama_Stimuli', 
    #                                           'Bgr_dire=Leftward_v=250', 
    #                                           'ET-Target_Num=1_W=5_H=5_V=1500_L=0-Traj=Ellipse_FPS=1000'),
    #                             is_video=False)
    # frame_reader = FrameIterator(os.path.join('D:/', 'STMD_Dataset', 'XS-VID', 
    #                                           'images', '13449-248520531_0'),
                                # is_video=False)
    # frame_reader = FrameIterator(os.path.join('C:/', 'Users', 'mings', 'OneDrive - University of Leicester', 
    #                                           'P2CODE-Mingshuo', 'videos', 'demo1-SeaDronesSee-696-1410.mp4'),
    #                             is_video=True)
    frame_reader = FrameIterator(os.path.join('D:/', 'STMD_Dataset', 'Real-World-Scence-Material', 'Chen', 'clean_GH010564.mp4'),
                                is_video=True)
    # visualizer
    visualizer = FrameVisualizer(window_name=model.__class__.__name__, 
                                 result_index_type="dots",
                                 win_height = frame_reader.img_height,
                                 win_width = frame_reader.img_width,
                                 conf_threshold=0)
    post_processor = PostProcessing(device=DEVICE, nms_radio=8, get_top_num=50)
    
    # Initialize
    # set the parameter list
    model.set_para()
    # print the parameter list
    model.print_para()
    # init
    model.setup()

    total_tunning_time = 0.0
    while True:
        color_img, gray_img, ret = frame_reader.get_next_frame(device=DEVICE)
        if not ret: break
            
        # Perform inference using the model
        time_start = time.time()
        result, run_time = inference(model, gray_img)

        dot_res = post_processor.process(result['response'], result['direction'])
        ret = visualizer.update(color_img, dot_res, process_time=run_time)
        if not ret: break

        total_tunning_time += time.time() - time_start

    print(f"Total time: {total_tunning_time:.4f} seconds, "\
          f"FPS: {frame_reader.current_index / total_tunning_time :.4f} frames/second")




if __name__ == "__main__":
    test()

    