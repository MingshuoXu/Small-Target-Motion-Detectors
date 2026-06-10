import os
import re
from pathlib import Path
import logging
from typing import Optional, List, Union, Tuple, Any
from functools import partial

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch


from .. import model
from .compute_module import PostProcessing, bbox_post_processing


# Get the full path of this file
filePath = os.path.realpath(__file__)
gitCodePath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(filePath))))
VID_DEFAULT_FOLDER = os.path.join(gitCodePath, 'demodata')
IMG_DEFAULT_FOLDER = os.path.join(VID_DEFAULT_FOLDER, 'imgstream')
# Add the path to the package containing the models
ALL_MODEL = model.__all__

logger = logging.getLogger(__name__)

class FrameIterator:
    """
    A flexible iterator class that can retrieve images frame-by-frame from a video file
    or from a sequence of numerically sorted image files.
    """

    def __init__(self, input_path: str, is_video: bool = True, is_silence: bool = True, device: str = 'cpu'):
        """
        Initialize the iterator.

        Parameters:
            input_path (str):
                - If is_video is True: full path to the video file.
                - If is_video is False: path to folder containing image sequence.
            is_video (bool): Specifies whether input is a video or image sequence.
            is_silence (bool): If True, suppresses standard informational output.
            device (str): Computation device for PyTorch tensors ('cpu', 'cuda', etc.).
        """
        self.input_path = input_path
        self.is_video = is_video
        self.is_silence = is_silence
        self.device = device  # 将 device 提升为类属性
        
        self.current_index = 0
        self.total_frames = 0
        self.is_open = False

        self.img_height, self.img_width = None, None
        self.cap = None
        self.image_files: List[str] = []

        if self.is_video:
            self._init_video_source()
        else:
            self._init_image_sequence_source()

    def _log(self, message: str, level: int = logging.INFO):
        """Helper to handle silenced logging"""
        if not self.is_silence or level >= logging.WARNING:
            logger.log(level, message)

    def _setup(self, current_index: int):
        """Jump to a specific frame index."""
        if current_index < 0 or (self.total_frames > 0 and current_index >= self.total_frames):
            logger.warning(f"Index {current_index} is out of bounds (0 - {self.total_frames-1}).")
            return

        self.current_index = current_index
        if self.is_video and self.cap:
            success = self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_index)
            if not success:
                logger.warning(f"Unable to set video frame position to {current_index}.")

    # --- Video processing logic ---
    def _init_video_source(self):
        if not os.path.isfile(self.input_path):
            logger.error(f"Video file not found: {self.input_path}")
            return

        self.cap = cv2.VideoCapture(self.input_path)
        if not self.cap.isOpened():
            logger.error(f"Unable to open video file: {self.input_path}")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.img_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.img_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.is_open = True
        
        self._log(f"Successfully opened video file. Total frames: {self.total_frames}")

    def _get_next_frame_from_video(self) -> Optional[np.ndarray]:
        if not self.is_open or not self.cap:
            return None
        
        ret, frame = self.cap.read()
        if ret:
            self.current_index += 1
            return frame
            
        self.release()  # Video reading completed or error occurred
        return None

    # --- Image sequence processing logic ---
    def _init_image_sequence_source(self):
        if not os.path.isdir(self.input_path):
            logger.error(f"Folder not found: {self.input_path}")
            return

        self.image_files = self._get_sorted_image_files(self.input_path)
        
        if not self.image_files:
            logger.error(f"No image files found in folder: {self.input_path}")
            return

        self.total_frames = len(self.image_files)
        self.is_open = True
        self._log(f"Successfully loaded image sequence. Total images: {self.total_frames}")

        # Read first image to get dimensions
        first_image = cv2.imread(self.image_files[0], cv2.IMREAD_COLOR)
        if first_image is not None:
            self.img_height, self.img_width = first_image.shape[:2]

    def _get_next_frame_from_sequence(self) -> Optional[np.ndarray]:
        while self.is_open and self.current_index < self.total_frames:
            file_path = self.image_files[self.current_index]
            
            # 【重要修复】无论是否读取成功，都必须 +1，否则读到坏图会死循环
            self.current_index += 1 
            
            frame = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if frame is not None:
                return frame
            else:
                logger.warning(f"Unable to read or decode image file: {file_path}")

        self.release()
        return None

    # --- Core interfaces ---
    def get_next_frame(self) -> Tuple[Optional[np.ndarray], Optional[torch.Tensor], bool]:
        """
        [Public interface] Get next image frame and its grayscale PyTorch tensor.

        Returns:
            Tuple[color_img, gray_tensor, is_valid]:
                - color_img: BGR image (NumPy array) or None
                - gray_tensor: Grayscale tensor shape (1, 1, H, W) or None
                - is_valid: Boolean indicating if retrieval was successful
        """
        color_img = self._get_next_frame_from_video() if self.is_video else self._get_next_frame_from_sequence()

        if color_img is None:
            return None, None, False

        gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        # 移除了中间不必要的变量，直接构造 tensor
        gray_tensor = torch.from_numpy(gray_img).to(device=self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0    

        return color_img, gray_tensor, True

    # --- Iterator & Context Manager Protocols ---
    def __iter__(self):
        return self

    def __next__(self) -> Tuple[np.ndarray, torch.Tensor]:
        color_img, gray_tensor, is_valid = self.get_next_frame()
        if not is_valid:
            raise StopIteration
        return color_img, gray_tensor

    def __enter__(self):
        """Enable context manager: `with FrameIterator(...) as it:`"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def release(self):
        if self.is_open:
            if self.is_video and self.cap is not None:
                self.cap.release()
            self.is_open = False
            self._log("Resources released.")

    def __del__(self):
        self.release()
        
    # --- Helpers ---
    @staticmethod
    def _natural_sort_key(s: str) -> List[Union[str, int]]:
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    def _get_sorted_image_files(self, folder_path: str) -> List[str]:
        """ Uses pathlib for faster and cleaner directory iteration. """
        valid_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif'}
        path = Path(folder_path)
        
        # iterdir() 比多次调用 glob 性能好得多
        files = [
            str(f) for f in path.iterdir() 
            if f.is_file() and f.suffix.lower() in valid_exts
        ]
        
        return sorted(files, key=lambda f: self._natural_sort_key(Path(f).name))
    

class FrameVisualizer:
    def __init__(self, window_name="Visualizer", 
                 win_width=None, win_height=None, 
                 is_headless=False,
                 conf_threshold=0.8 # 阈值参数
                 ): 
        """
        初始化可视化器
        :param conf_threshold: 可视化过滤的相对阈值 (0.0 ~ 1.0)
        """
        self.window_name = window_name
        self.win_width = win_width or 800
        self.win_height = win_height or 600
        self.is_headless = is_headless
        self.conf_threshold = conf_threshold
        
        self.save_output = False
        self.video_writer = None # 显式初始化为 None
        self.paused = False
        
        self._setup_window()
    
    def _setup_window(self):
        """初始化窗口"""
        if self.is_headless:
            return
        cv2.namedWindow(self.window_name, cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(self.window_name, self.win_width, self.win_height)

    def setup_video_writer(self, output_path, fps=30, width=None, height=None):
        """初始化视频写入器 (建议外部显式调用)"""
        # 确保目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        width = width if width is not None else self.win_width
        height = height if height is not None else self.win_height
        
        # 常用 mp4v 兼容性较好
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        self.save_output = True
        print(f">>> Video writer initialized: {output_path}")

    def update(self, frame, result=None, direction=None, annotation=None, show_str=None) -> bool:
        if frame is None:
            return False

        # --- 绘制逻辑 ---
        # 即使 result 是空的，只要不为 None 也可以处理
        if result is not None:
            if result.dim() == 4:
                self._draw_matrix(frame, result, direction, self.conf_threshold)
            elif result.shape[1] == 4: 
                result = result.cpu().numpy() if isinstance(result, torch.Tensor) else result
                self._draw_dots(frame, result, self.conf_threshold)
            elif result.shape[1] == 5: 
                self._draw_bbox(frame, result, self.conf_threshold, annotation)
            device_str = f'{result.device}'
        else:
            device_str = 'Time'
        # --- 信息显示 ---
        if show_str is not None and show_str != '':
            cv2.putText(frame, str(show_str),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2, cv2.LINE_AA)
            
        # --- 视频保存 (安全检查) ---
        if self.save_output and self.video_writer is not None:
            self.video_writer.write(frame)

        # --- Headless 快速返回 ---
        if self.is_headless:
            return True
        
        # --- 窗口显示与按键控制 ---
        cv2.imshow(self.window_name, frame)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord('q'): # Esc
                return False

            if key == 32: # Space
                self.paused = not self.paused
                print(f">>> State: {'Paused' if self.paused else 'Running'}")
                if not self.paused: break
                continue

            if key == ord('n'): # Next
                self.paused = True
                break 

            if not self.paused:
                break
        
        return True

    def close(self):
        if not self.is_headless:
            cv2.destroyWindow(self.window_name)
        if self.video_writer is not None:
            self.video_writer.release()

    @staticmethod
    def _draw_arrows(frame, x_coords, y_coords, directions, length=15):
        """
        :param x_coords: X 坐标数组 (Cols)
        :param y_coords: Y 坐标数组 (Rows)
        :param directions: 方向角数组 (弧度)
        """
        if len(x_coords) == 0: return

        cos_d = np.cos(directions)
        sin_d = np.sin(directions)

        # Zip 里的顺序明确为: x, y, cos, sin
        for x, y, c, s in zip(x_coords, y_coords, cos_d, sin_d):
            # 必须转为 int，因为 cv2 坐标不支持 float
            start_pt = (int(x), int(y))
            end_pt = (int(x + length * c), int(y - length * s))
            
            cv2.arrowedLine(frame, start_pt, end_pt,
                            color=(0, 0, 255), thickness=1, 
                            tipLength=0.3, line_type=cv2.LINE_AA)

    @staticmethod
    def _draw_matrix(frame, matrix, direction_map, threshold):
        """处理 Matrix 格式 (Heatmap)"""
        if torch.max(matrix) <= 0: return

        # np.where 返回 (rows, cols) 即 (y, x)
        _, _, rows, cols = torch.where(matrix > threshold)
        rows = rows.cpu().numpy()
        cols = cols.cpu().numpy()
        
        # 画点
        for r, c in zip(rows, cols):
            # cv2 坐标是 (x, y) -> (col, row)
            cv2.drawMarker(frame, (c, r), color=(0, 0, 255), 
                           markerType=cv2.MARKER_STAR, markerSize=5, thickness=1)

        # 画箭头
        if direction_map is not None and len(rows) > 0:
            # 确保 direction_map 维度匹配，这里假设是同样大小的矩阵
            valid_dirs = direction_map[0, 0, rows, cols]
            
            # 过滤 NaN
            valid_mask = ~np.isnan(valid_dirs)
            if np.any(valid_mask):
                # 传入 _draw_arrows 的必须是 (x, y) 对应 (cols, rows)
                FrameVisualizer._draw_arrows(frame, 
                                             x_coords=cols[valid_mask], 
                                             y_coords=rows[valid_mask], 
                                             directions=valid_dirs[valid_mask])

    @staticmethod
    def _draw_dots(frame, response, threshold):
        """处理 Dots 格式: [[x, y, score, dir], ...]"""
        if len(response) == 0: return

        # 假设格式: Col 0=x, Col 1=y, Col 2=score
        # 安全过滤
        scores = response[:, 2]

        mask = scores > threshold
        filtered = response[mask]

        if len(filtered) == 0: return

        xs = filtered[:, 0]
        ys = filtered[:, 1]

        for x, y in zip(xs, ys):
            cv2.drawMarker(frame, (int(x), int(y)), color=(0, 0, 255), 
                           markerType=cv2.MARKER_STAR, markerSize=5, thickness=1)

        # 处理方向 (假设 Col 3 是方向)
        if response.shape[1] > 3:
            dirs = filtered[:, 3]
            valid_mask = ~np.isnan(dirs)
            FrameVisualizer._draw_arrows(frame, 
                                         x_coords=xs[valid_mask], 
                                         y_coords=ys[valid_mask], 
                                         directions=dirs[valid_mask])

    @staticmethod
    def _draw_bbox(frame, response, threshold, annotation=None):
        """处理 BBox 格式: [[x1, y1, x2, y2, score, dir], ...]"""
        if response.size == 0: return

        # 1. 提前过滤：先做 Mask 过滤，减少后续转换的数据量
        response = response.cpu().numpy() if isinstance(response, torch.Tensor) else response
        mask = response[:, 4] > threshold
        filtered_res = response[mask]
        if filtered_res.size == 0: return

        # 2. 批量转换类型
        # 只转换坐标部分，避免对整个 response 进行转换
        boxes = filtered_res[:, :4].astype(np.int32)
        
        # 3. 提取方向（如果有）
        # 优化点：直接从过滤后的结果拿第 5 列，避免多次索引 filtered
        has_dir = filtered_res.shape[1] > 5
        
        # 4. 优化循环逻辑：将判断移出循环
        if annotation is not None:
            filtered_anno = np.asanyarray(annotation)[mask]
            for (x1, y1, x2, y2), anno in zip(boxes, filtered_anno):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, str(anno), (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)

        # 5. 绘制方向向量（矢量化计算中心点）
        if has_dir:
            dirs = filtered_res[:, 5]
            v_mask = ~np.isnan(dirs)
            if np.any(v_mask):
                v_boxes = boxes[v_mask]
                # 使用位移运算或更快的加法，并保持 float 计算中心点
                # 这里的 // 2 直接得到整数坐标，方便画图
                c_xs = (v_boxes[:, 0] + v_boxes[:, 2]) // 2
                c_ys = (v_boxes[:, 1] + v_boxes[:, 3]) // 2
                FrameVisualizer._draw_arrows(frame, c_xs, c_ys, dirs[v_mask])


class ModelSelectorGUI:
    def __init__(self, root):
        self.root = root

    def create_gui(self, modelList):
        self.modelLabel = ttk.Label(self.root, text="Select A Model:", width = 15)
        self.modelLabel.grid(row=0, column=0, padx=10, pady=10)

        self.modelCombobox = ttk.Combobox(self.root, values=modelList, width = 25)
        self.modelCombobox.current(11)
        self.modelCombobox.grid(row=0, column=1, columnspan=2, pady=10, sticky='w')
        

class InputSelectorGUI:
    def __init__(self, root):
        self.root = root

        self.vidElement = {}
        self.imgElement = {}

        self.imgSelectFolder = None

        self.inputType = None
        self.startFrame = None
        self.endFrame = None
        self.video_file = None
        self.check = False
        self.vidName = None
        self.startFolder = None
        self.startImgName = None
        self.endFolder = None
        self.endImgName = None

    def create_gui(self):
        self.inputTypeLabel = ttk.Label(self.root, text="Input Type:", width = 15)
        self.inputTypeLabel.grid(row=1, column=0, padx=10, pady=10, sticky='w')

        self.selectedOption = tk.IntVar(value=0)

        self.imgLabel = ttk.Radiobutton(self.root, 
                                        text='Image Sequence', 
                                        variable=self.selectedOption,
                                        value=2, 
                                        command=self.select_imgstream)
        self.imgLabel.grid(row=1, column=2, padx=10, pady=10, sticky="w")
        
        self.vidLabel = ttk.Radiobutton(self.root, 
                                        text='Video', 
                                        variable=self.selectedOption,
                                        value=1, 
                                        command=self.select_vidstream)
        self.vidLabel.grid(row=1, column=1, padx=10, pady=10, sticky="w")

    def select_vidstream(self):
        self.imgSelectFolder = None
        self.startImgName = None
        self.endImgName = None
        for element in self.imgElement.values():
            element.destroy()

        self.vidElement['lblVidIndicate'] = ttk.Label(self.root, text= 'Video\'s path:')
        self.vidElement['lblVidIndicate'].grid(row=2, column=1, padx=10, pady=30, sticky='w')
        self.vidElement['lblVidPath'] = ttk.Label(self.root, 
                                           text="Waiting for the selection",
                                           wraplength=220
                                           )
        self.vidElement['lblVidPath'].grid(row=2, column=2, padx=10, pady=10, sticky='w')
        
        self.vidElement['btn'] = ttk.Button(self.root, text="Select a video", command=self._clicked_vid)
        self.vidElement['btn'].grid(row=3, column=2, padx=10, pady=10, sticky='w')
        
    def _clicked_vid(self):
        self.vidName = filedialog.askopenfilenames(initialdir=VID_DEFAULT_FOLDER)
        self.vidName = self.vidName[0]
        self.vidElement['lblVidPath'].config(text=self.vidName)

    def select_imgstream(self):
        self.vidName = None
        for element in self.vidElement.values():
            element.destroy()

        self.imgElement['lblFolder'] = ttk.Label(self.root, text="Image's folder: ")
        self.imgElement['lblFolder'].grid(row=2, column=1, padx=10, pady=10, sticky='w')
        self.imgElement['lblFolderName'] = ttk.Label(self.root, text="Waiting for the selection", wraplength=220)
        self.imgElement['lblFolderName'].grid(row=2, column=2, padx=10, pady=30, sticky='w')

        self.imgElement['btnStart'] = ttk.Button(self.root, text="Select start frame", command=self._clicked_start_img)
        self.imgElement['btnStart'].grid(row=3, column=1,  padx=10, pady=10, sticky='w')
        self.imgElement['lblStartImg'] = ttk.Label(self.root, text=self.startImgName)
        self.imgElement['lblStartImg'].grid(row=3, column=2, padx=10, pady=10, sticky='w')

        self.imgElement['btnEnd'] = ttk.Button(self.root, text="Select end frame", command=self._clicked_end_img)
        self.imgElement['btnEnd'].grid(row=4, column=1,  padx=10, pady=10, sticky='w')
        self.imgElement['lblEndImg'] = ttk.Label(self.root, text=self.endImgName)
        self.imgElement['lblEndImg'].grid(row=4, column=2, padx=10, pady=10, sticky='w')
        
    def _clicked_start_img(self):
        startImgFullPath = filedialog.askopenfilenames(
            initialdir=IMG_DEFAULT_FOLDER if self.imgSelectFolder is None else self.imgSelectFolder)
        self.startFolder, self.startImgName = os.path.split(startImgFullPath[0])
        if self.endFolder is not None:
            if os.path.basename(self.startFolder) == os.path.basename(self.endFolder):
                if self.endImgName is not None:
                    if check_same_ext_name(self.startImgName, self.endImgName):
                        self.check = True
                    else:
                        messagebox.showinfo("Message title", "Start image has a different extension than end image.")
            else:
                messagebox.showinfo("Message title", "The image stream must be in the same folder!")

        self.imgSelectFolder = self.startFolder
        self.imgElement['lblFolderName'].config(text=self.imgSelectFolder)

        self.imgElement['lblStartImg'].config(text=self.startImgName)

    def _clicked_end_img(self):
        endImgFullPath = filedialog.askopenfilenames(
            initialdir=IMG_DEFAULT_FOLDER if self.imgSelectFolder is None else self.imgSelectFolder)
        self.endFolder , self.endImgName = os.path.split(endImgFullPath[0])

        if self.startFolder is not None:
            if os.path.basename(self.endFolder) == os.path.basename(self.startFolder):
                if self.startImgName is not None:
                    if check_same_ext_name(self.startImgName, self.endImgName):
                        self.check = True
                    else:
                        messagebox.showinfo("Message title", "Start image has a different extension than end image.")
            else:
                messagebox.showinfo("Message title", "The image stream must be in the same folder!")

                
        self.imgSelectFolder = self.endFolder
        self.imgElement['lblFolderName'].config(text=self.imgSelectFolder)

        self.imgElement['lblEndImg'].config(text=self.endImgName)


class PostProcessingSelectorGUI:
    def __init__(self, root):
        self.root = root
        self.output_type = "dot"
        self.show_threshold = 0.0
        self.top_num = 1

        self.outputTypeLabel = ttk.Label(self.root, text="Output Type:", width = 15)
        self.outputTypeLabel.grid(row=5, column=0, padx=10, pady=10)

        self.selectedOption = tk.IntVar(value=2)

        self.dotLabel = ttk.Radiobutton(self.root, 
                                        text='STMD Output', 
                                        variable=self.selectedOption,
                                        value=2, 
                                        command=self.select_dot)
        self.dotLabel.grid(row=5, column=1, padx=10, pady=10, sticky="w")
        
        self.bboxLabel = ttk.Radiobutton(self.root, 
                                        text='Bbox Output', 
                                        variable=self.selectedOption,
                                        value=1, 
                                        command=self.select_bbox)
        self.bboxLabel.grid(row=5, column=2, padx=10, pady=10, sticky="w")

        self.showThresholdLabel = ttk.Label(self.root, text="Threshold:", width=10)
        self.showThresholdLabel.grid(row=6, column=1, padx=10, pady=10, sticky='w')

        self.showThresholdVar = tk.StringVar(value="0")
        self.showThresholdVar.trace_add('write', self.update_show_threshold)
        self.showThresholdEntry = ttk.Entry(self.root, textvariable=self.showThresholdVar, width=5)
        self.showThresholdEntry.grid(row=6, column=2, padx=10, pady=10, sticky='w')

        self.getTopNumLabel = ttk.Label(self.root, text="Top Num:", width=10)
        self.getTopNumLabel.grid(row=7, column=1, padx=10, pady=10, sticky='w')

        self.getTopNumVar = tk.StringVar(value="1")
        self.getTopNumVar.trace_add('write', self.update_top_num)
        self.getTopNumEntry = ttk.Entry(self.root, textvariable=self.getTopNumVar, width=5)
        self.getTopNumEntry.grid(row=7, column=2, padx=10, pady=10, sticky='w')
        
        self.select_dot()

    def select_dot(self):
        self.selectedOption.set(2)
        self.output_type = "dot"

    def select_bbox(self):
        self.selectedOption.set(1)
        self.output_type = "bbox"

    def get_post_processing(self):
        if self.output_type == "dot":
            return PostProcessing(get_top_num = self.top_num)
        elif self.output_type == "bbox":
            return bbox_post_processing(self.top_num)
        else:
            raise ValueError(f"Unknown output type: {self.output_type}")

    def update_show_threshold(self, *args):
        try:
            value = float(self.showThresholdVar.get())
        except ValueError:
            value = 0.0
        self.show_threshold = min(max(value, 0.0), 1.0)  # 确保在 [0.0, 1.0] 范围内
        
    def update_top_num(self, *args):
        try:
            value = int(self.getTopNumVar.get())
        except ValueError:
            # 如果输入为空或者包含非数字字符，赋予一个默认值
            value = 0

        self.top_num = max(value, 1)  # 确保 top_num 至少为 1


class DeviceSelectorGUI:
    def __init__(self, root):
        self.root = root
        self.device = "cpu"

        self.deviceLabel = ttk.Label(self.root, text="Select Device:", width=15)
        self.deviceLabel.grid(row=8, column=0, padx=10, pady=10)

        self.selectedOption = tk.IntVar(value=1)

        if torch.cuda.is_available():
            self.selectedOption.set(2)
            self.device = "cuda"
        
        self.device_frame = ttk.Frame(self.root)
        self.device_frame.grid(row=8, column=1, columnspan=2, padx=10, pady=10, sticky="w")

        self.cpuLabel = ttk.Radiobutton(self.device_frame, 
                                text='CPU', 
                                variable=self.selectedOption,
                                value=1, 
                                command=self.select_cpu)
        # 使用 pack(side="left") 可以让它们在 frame 内从左到右水平挨着排列
        self.cpuLabel.pack(side="left", padx=(0, 20)) # 右侧留 20px 间距

        self.cudaLabel = ttk.Radiobutton(self.device_frame, 
                                        text='CUDA', 
                                        variable=self.selectedOption,
                                        value=2, 
                                        command=self.select_cuda)
        self.cudaLabel.pack(side="left", padx=(0, 20))

        self.mpsLabel = ttk.Radiobutton(self.device_frame, 
                                        text='MPS', 
                                        variable=self.selectedOption,
                                        value=3, 
                                        command=self.select_mps)
        self.mpsLabel.pack(side="left")

    def select_cpu(self):
        self.selectedOption.set(1)
        self.device = "cpu"

    def select_cuda(self):
        if torch.cuda.is_available():
            self.selectedOption.set(2)
            self.device = "cuda"
        else:
            messagebox.showinfo("Message title", "CUDA is not available. Please select CPU.")
            self.select_cpu()

    def select_mps(self):
        if torch.backends.mps.is_available():
            self.selectedOption.set(3)
            self.device = "mps"
        else:
            messagebox.showinfo("Message title", "Metal Performance Shaders (MPS) is not available. Please select CPU.")
            self.select_cpu()


class XTTMP_GUI:
    def __init__(self, root):
        self.root = root

        windowHeight = 550
        windowWidth = 510
        
        startHeight = (root.winfo_screenheight() - windowHeight) // 2
        startWidth = (root.winfo_screenwidth() - windowWidth) // 2

        self.root.geometry('{}x{}+{}+{}'.format(windowWidth, windowHeight, startWidth, startHeight))
        self.root.title("Small target motion detector - Runner")
        self._set_window_icon()
        
        self.objModelSelector = ModelSelectorGUI(root)
        self.objPostProcessingSelector = PostProcessingSelectorGUI(root)
        self.objInputSelector = InputSelectorGUI(root)
        self.objDeviceSelector = DeviceSelectorGUI(root)
        
        
        self.btnStepping = ttk.Button(self.root, text="Stepping", command=self._stepping)
        self.isStepping = False
        self.btnStepping.grid(row=9, column=2, padx=10, pady=10, sticky='e')
        self.btnRun = ttk.Button(self.root, text="Run", command=self._run)
        self.btnRun.grid(row=10, column=2, padx=10, pady=10, sticky='e')

    def create_gui(self):
        self.objModelSelector.create_gui(ALL_MODEL)
        self.objInputSelector.create_gui()

        self.root.mainloop()

        if self.objInputSelector.selectedOption.get() == 1:  
            return (self.modelName, self.vidName, None, self.isStepping, 
                    self.objDeviceSelector.device, self.objPostProcessingSelector.get_post_processing(),
                    self.objPostProcessingSelector.show_threshold)
        elif self.objInputSelector.selectedOption.get() == 2:
            return (self.modelName, self.startImgName, self.endImgName, self.isStepping,
                     self.objDeviceSelector.device, self.objPostProcessingSelector.get_post_processing(),
                    self.objPostProcessingSelector.show_threshold)

    def _run(self):
        self.modelName = self.objModelSelector.modelCombobox.get()
        if self.modelName not in ALL_MODEL:
            messagebox.showinfo("Message title", "Please select a STMD-based model!")
            return

        if self.objInputSelector.selectedOption.get() == 1:
            if self.objInputSelector.vidName is not None:
                self.vidName = self.objInputSelector.vidName
                self.root.destroy()
            else:
                messagebox.showinfo("Message title", "Please select a video")
        elif self.objInputSelector.selectedOption.get() == 2:
            if self.objInputSelector.startImgName is None:
                messagebox.showinfo("Message title", "Please select start frame!")
                return
            
            if self.objInputSelector.endImgName is None:
                messagebox.showinfo("Message title", "Please select end frame!")
                return

            if self.objInputSelector.check:
                self.startImgName = os.path.join(self.objInputSelector.imgSelectFolder, 
                                        self.objInputSelector.startImgName)
                self.endImgName = os.path.join(self.objInputSelector.imgSelectFolder, 
                                        self.objInputSelector.endImgName)
                self.root.destroy()
            else:
                messagebox.showinfo("Message title", "The image stream must be in the same folder!")
        else:
            messagebox.showinfo("Message title", "Please select input")
    
    def _stepping(self):
        self.isStepping = True
        self._run()

    def _set_window_icon(self):
        icon_path = Path(__file__).resolve().with_name('stmd.ico')
        if not icon_path.is_file():
            logger.warning('Window icon not found: %s', icon_path)
            return

        try:
            self.root.iconbitmap(str(icon_path))
        except tk.TclError as exc:
            logger.warning('Unable to set window icon %s: %s', icon_path, exc)


def check_same_ext_name(startImgName, endImgName):
    _, ext1 = os.path.splitext(startImgName)
    _, ext2 = os.path.splitext(endImgName)
    # Check if the extensions of the start and end images are the same
    if ext1 != ext2:
        return False
    else:
        return True
    

