# demo_imgstream
import os
import sys
import time

import torch

filePath = os.path.realpath(__file__)
project_path = os.path.dirname(os.path.dirname(os.path.dirname(filePath)))
gitCodePath = os.path.dirname(project_path)
sys.path.append(project_path)

from xttmp.util.iostream import FrameIterator, FrameVisualizer
from xttmp.api import instancing_model  # type: ignore
from xttmp.util.compute_module import PostProcessing  # type: ignore


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = 'vSTMD_F'
INPUT_PATH = os.path.join(gitCodePath, 'example-data', 'imgstream')
SHOW_THRESHOLD = 0.2
GET_TOP_NUM = 20


def main():
    model = instancing_model(MODEL_NAME, device=DEVICE)

    frame_reader = FrameIterator(INPUT_PATH, is_video=False, device=DEVICE)
    visualizer = FrameVisualizer(
        window_name=model.__class__.__name__,
        result_index_type='dots',
        win_height=frame_reader.img_height,
        win_width=frame_reader.img_width,
        conf_threshold=SHOW_THRESHOLD,
    )
    post_processor = PostProcessing(device=DEVICE, nms_radio=8, get_top_num=GET_TOP_NUM)

    total_time = 0.0

    try:
        for color_img, gray_tensor in frame_reader:
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            time_start = time.time()

            result = model(gray_tensor)

            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            run_time = time.time() - time_start

            dot_res = post_processor(result['response'], result.get('direction'))
            if not visualizer.update(color_img, dot_res, process_time=run_time):
                break

            total_time += run_time

        if total_time > 0:
            print(f"Total time: {total_time:.4f} seconds, "
                  f"FPS: {frame_reader.current_index / total_time :.4f} frames/second")
    finally:
        visualizer.close()
        frame_reader.release()


if __name__ == "__main__":
    main()