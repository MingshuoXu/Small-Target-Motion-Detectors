import os
import sys
import torch
import time

filePath = os.path.realpath(__file__)
project_path = os.path.dirname(os.path.dirname(os.path.dirname(filePath)))
sys.path.append(os.path.join(project_path, 'src'))
from xttmp.util.iostream import FrameIterator, FrameVisualizer
from xttmp.util.compute_module import PostProcessing
from xttmp.model.backbone import ESTMD

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

import logging
logging.basicConfig(level=logging.INFO)

def test():
    # model
    model = ESTMD()
    # model.set_para() # set the parameter list
    # model.setup() # initialize the model if you have set the parameter list
    model.print_para() # print the parameter list
    model.to(device=DEVICE)
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
    # frame_reader = FrameIterator(os.path.join('D:/', 'STMD_Dataset', 'Real-World-Scence-Material', 'Chen', 'clean_GH010564.mp4'),
    #                             is_video=True)
    frame_reader = FrameIterator(os.path.join(project_path, 'example-data', 'simulatedVideo0_orignal_1000Hz.mp4'),
                                device=DEVICE, is_video=True)
    # visualizer
    visualizer = FrameVisualizer(window_name=model.__class__.__name__, 
                                 win_height = frame_reader.img_height,
                                 win_width = frame_reader.img_width,
                                 conf_threshold=0)
    post_processor = PostProcessing(nms_radio=8, get_top_num=1)
    
    total_tunning_time = 0.0
    for color_img, gray_tensor in frame_reader:
            
        # Perform inference using the model
        if DEVICE == 'cuda':
            torch.cuda.synchronize()  # Ensure all CUDA operations are complete before starting the timer
        time_start = time.time()
        result = model(gray_tensor)

        if DEVICE == 'cuda':
            torch.cuda.synchronize()  # Ensure all CUDA operations are complete before stopping the timer
        run_time = time.time() - time_start

        dot_res = post_processor.process(result['response'])
        _show_str = f"{DEVICE.upper()}: {run_time*1000:.1f} ms"
        ret = visualizer.update(color_img, dot_res, show_str=_show_str)
        if not ret: break

        total_tunning_time += run_time

    print(f"Total time: {total_tunning_time:.4f} seconds, "\
          f"FPS: {frame_reader.current_index / total_tunning_time :.4f} frames/second")



if __name__ == "__main__":
    test()

    