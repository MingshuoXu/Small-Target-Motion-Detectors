import os
import sys
import time
from typing import Optional

import tkinter as tk
import torch

file_path = os.path.realpath(__file__)
project_path = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
repo_root = os.path.dirname(project_path)
sys.path.append(project_path)

from xttmp.util.iostream import ModelAndInputSelectorGUI, FrameIterator, FrameVisualizer
from xttmp.api import instancing_model  # type: ignore
from xttmp.util.compute_module import PostProcessing  # type: ignore


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class StmdGuiSingleProcess:
    def __init__(self, device: str = DEVICE, show_threshold: float = 0.2, get_top_num: int = 20):
        self.device = device
        self.show_threshold = show_threshold
        self.get_top_num = get_top_num
        self.ModelAndInputSelectorGUI = ModelAndInputSelectorGUI
        self.FrameIterator = FrameIterator
        self.FrameVisualizer = FrameVisualizer
        self.PostProcessing = PostProcessing
        self.instancing_model = instancing_model

    def _get_user_input(self):
        root = tk.Tk()
        try:
            gui = self.ModelAndInputSelectorGUI(root)
            return gui.create_gui()
        finally:
            try:
                root.destroy()
            except tk.TclError:
                pass

    def _create_frame_reader(self, opt1: str, opt2: Optional[str]):
        if opt2 is None:
            return self.FrameIterator(opt1, is_video=True, device=self.device)

        reader = self.FrameIterator(os.path.dirname(opt1), is_video=False, device=self.device)
        start_name = os.path.basename(opt1)
        end_name = os.path.basename(opt2)

        start_index = next((i for i, path in enumerate(reader.image_files)
                            if os.path.basename(path) == start_name), None)
        end_index = next((i for i, path in enumerate(reader.image_files)
                          if os.path.basename(path) == end_name), None)

        if start_index is None or end_index is None:
            raise ValueError("Selected image range could not be located in the folder.")

        if start_index > end_index:
            start_index, end_index = end_index, start_index

        reader._setup(start_index)
        reader.total_frames = end_index + 1
        return reader

    def run(self):
        reader = None
        visualizer = None
        try:
            user_input = self._get_user_input()
            if not user_input:
                return

            model_name, opt1, opt2, is_stepping = user_input
            reader = self._create_frame_reader(opt1, opt2)
            model = self.instancing_model(model_name, device=self.device)
            post_processor = self.PostProcessing(
                device=self.device,
                nms_radio=8,
                get_top_num=self.get_top_num,
            )

            visualizer = self.FrameVisualizer(
                window_name=model.__class__.__name__,
                result_index_type='dots',
                win_height=reader.img_height,
                win_width=reader.img_width,
                conf_threshold=self.show_threshold,
            )
            if is_stepping:
                visualizer.paused = True

            total_time = 0.0

            while True:
                color_img, gray_tensor, is_valid = reader.get_next_frame()
                if not is_valid:
                    break

                if self.device == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()

                result = model(gray_tensor)

                if self.device == 'cuda':
                    torch.cuda.synchronize()
                run_time = time.time() - start_time

                dot_res = post_processor(result['response'], result.get('direction'))
                if not visualizer.update(color_img, dot_res, process_time=run_time):
                    break

                total_time += run_time

            if total_time > 0:
                print(f"Total time: {total_time:.4f} seconds, "
                      f"FPS: {reader.current_index / total_time :.4f} frames/second")

        finally:
            if visualizer is not None:
                visualizer.close()
            if reader is not None:
                reader.release()


def main(show_threshold: float = 0.2, get_top_num: int = 20):
    app = StmdGuiSingleProcess(device=DEVICE, show_threshold=show_threshold, get_top_num=get_top_num)
    app.run()


if __name__ == '__main__':
    main()