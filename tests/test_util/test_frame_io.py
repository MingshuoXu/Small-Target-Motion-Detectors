import os
import sys

import cv2
import numpy as np

filePath = os.path.realpath(__file__)
project_path = os.path.dirname(os.path.dirname(os.path.dirname(filePath)))
sys.path.append(os.path.join(project_path, 'src'))
from xttmp.util.iostream import FrameIterator, FrameVisualizer

# Create test object


def test():
    frame_reader = FrameIterator(os.path.join(project_path, 'example-data', 'RIST_GX010290_orignal_240Hz.mp4'), is_video=True)

    visualizer = FrameVisualizer(window_name="Test Visualizer", result_index_type="bbox",
                                 win_height = frame_reader.img_height,
                                 win_width = frame_reader.img_width)

    is_running = True
    while is_running:
        gray_img, color_img, ret = frame_reader.get_next_frame()
        if not ret:
            break

        # Dummy result for testing
        dummy_result = np.array([[50, 50, 200, 200, 0.9],])  # x1, y1, x2, y2, confidence

        is_running = visualizer.update(color_img, dummy_result)




if __name__ == "__main__":
    test()

