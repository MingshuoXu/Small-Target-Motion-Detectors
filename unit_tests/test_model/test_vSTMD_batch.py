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
from xttmp.api import instancing_model # type: ignore
from xttmp.util.compute_module import PostProcessing # type: ignore


# DEVICE = 'cpu' # 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def test():
    # model
    model = instancing_model('vSTMD', device=DEVICE) 
    model.to(device=DEVICE)
    # input
    # frame_reader = FrameIterator(os.path.join(project_path, 'example-data', 'RIST_GX010290_orignal_240Hz.mp4'), is_video=True)
    frame_reader = FrameIterator(os.path.join('D:/', 'STMD_Dataset', 'vSTMD_Panorama_Stimuli', 
                                              'Bgr_dire=Leftward_v=250', 
                                              'ET-Target_Num=1_W=5_H=5_V=1500_L=0-Traj=Ellipse_FPS=1000'),
                                device=DEVICE, is_video=False)
    

    total_tunning_time = 0.0
    batch_size = 16
    i = 0
    for color_img, gray_tensor in frame_reader:
        
        if i < batch_size-1:
            if i == 0:
                input_torch = gray_tensor
            else:
                input_torch = torch.cat((input_torch, gray_tensor), dim=0)
            i += 1
            continue

        input_torch = torch.cat((input_torch, gray_tensor), dim=0)

        # Perform inference using the model
        time_start = time.time()
        results = model.forward(input_torch)
        torch.cuda.synchronize() if DEVICE == 'cuda' else None

        total_tunning_time += time.time() - time_start

        i = 0

    print(f"Total time: {total_tunning_time:.4f} seconds, "\
          f"FPS: {frame_reader.current_index / total_tunning_time :.4f} frames/second")




if __name__ == "__main__":
    test()

    