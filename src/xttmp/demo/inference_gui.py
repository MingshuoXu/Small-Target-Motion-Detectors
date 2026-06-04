import os
import sys
import logging
import time
from typing import Optional


import tkinter as tk
import torch

file_path = os.path.realpath(__file__)
py_pkg_path = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
sys.path.append(py_pkg_path)

try:
    from xttmp.util.iostream import ( # type: ignore
                    ModelAndInputSelectorGUI,
                    FrameIterator,
                    FrameVisualizer,
                )
    from xttmp.util.compute_module import PostProcessing # type: ignore
    from xttmp.api import ( # type: ignore
        instancing_model,
    ) 
except ImportError as e:
    raise ImportError("Failed to import required modules. "
                      "Ensure that the 'xttmp' package is correctly installed.") from e

# configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StmdGui:
    def __init__(self, device='cpu', show_threshold: float = 0.8, get_top_num: int = 1):
        """ Initialize STMD GUI """
        self.device = device
        self.show_threshold = show_threshold
        self.get_top_num = get_top_num
        self.ModelAndInputSelectorGUI = ModelAndInputSelectorGUI
        self.FrameIterator = FrameIterator
        self.FrameVisualizer = FrameVisualizer
        self.PostProcessing = PostProcessing
        self.instancing_model = instancing_model


    def _get_user_input(self) -> tuple:
        """ get user input """
        root = tk.Tk()
        try:
            gui = self.ModelAndInputSelectorGUI(root)
            return gui.create_gui()
        finally:
            # FIX: 安全销毁逻辑
            # gui.create_gui() 可能已经销毁了窗口（例如用户点击了确认按钮后代码内部调用了 destroy）
            # 所以这里包裹一个 try-except，如果窗口已不在，直接忽略错误。
            try:
                root.destroy()
            except tk.TclError:
                pass

    def _create_frame_reader(self, opt1: str, opt2: Optional[str]):
        """Create a frame reader for a video file or an image sequence."""
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
        """ run video processor"""
        reader = None
        visualizer = None
        try:
            user_input = self._get_user_input()
            if not user_input:
                logger.info("User cancelled input.")
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
                window_name=model_name,
                result_index_type="dots",
                win_width=reader.img_width,
                win_height=reader.img_height,
                conf_threshold=self.show_threshold,
            )
            if is_stepping:
                visualizer.paused = True

            while True:
                color_img, gray_tensor, is_valid = reader.get_next_frame()
                if not is_valid:
                    break

                if self.device != 'cpu' and torch.cuda.is_available():
                    torch.cuda.synchronize()
                time_start = time.perf_counter()
                result = model(gray_tensor)
                if self.device != 'cpu' and torch.cuda.is_available():
                    torch.cuda.synchronize()
                run_time = time.perf_counter() - time_start

                dots = post_processor(result['response'], result.get('direction'))
                if not visualizer.update(color_img, result=dots, process_time=run_time):
                    break

        except Exception as e:
            logger.error(f"Main process error: {str(e)}")
        finally:
            logger.info("Cleaning up resources...")
            if visualizer is not None:
                visualizer.close()
            if reader is not None:
                reader.release()
            logger.info("Shutdown completed")

def main(show_threshold: float = 0, get_top_num: int = 10):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    obj = StmdGui(DEVICE, show_threshold = show_threshold, get_top_num = get_top_num)
    obj.run()

if __name__ == "__main__":
    main(get_top_num = 20)