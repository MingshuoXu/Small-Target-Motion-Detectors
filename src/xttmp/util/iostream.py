import glob
import os
import re
import sys
import time
from typing import List, Optional, Union



import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch

from .compute_module import AreaNMS
from .. import model


# Get the full path of this file
filePath = os.path.realpath(__file__)
gitCodePath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(filePath))))
VID_DEFAULT_FOLDER = os.path.join(gitCodePath, 'demodata')
IMG_DEFAULT_FOLDER = os.path.join(VID_DEFAULT_FOLDER, 'imgstream')
# Add the path to the package containing the models
ALL_MODEL = model.__all__



class FrameIterator:
    """
    A flexible iterator class that can retrieve images frame-by-frame from a video file
    or from a sequence of numerically sorted image files.
    """

    def __init__(self, input_path: str, is_video: bool = True, is_silence: bool = True):
        """
        Initialize the iterator.

        Parameters:
            input_path (str):
                - If is_video is True: full path to the video file.
                - If is_video is False: path to folder containing image sequence.
            is_video (bool): Specifies whether input is a video or image sequence.
        """
        self.input_path = input_path
        self.is_video = is_video
        self.current_index = 0
        self.total_frames = 0
        self.is_open = False
        self.is_silence = is_silence  

        self.img_height, self.img_width = None, None

        if self.is_video:
            self._init_video_source()
        else:
            self._init_image_sequence_source()

    def _setup(self, current_index: int):
        self.current_index = current_index
        if self.is_video:
            success = self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_index)
            if not success:
                print(f"Warning: Unable to set video frame position to {current_index}.")


    # --- Video processing logic ---
    def _init_video_source(self):
        """ Initialize video file reading. """
        if not os.path.isfile(self.input_path):
            print(f"Error: Video file not found: {self.input_path}")
            return

        self.cap = cv2.VideoCapture(self.input_path)
        if not self.cap.isOpened():
            print(f"Error: Unable to open video file: {self.input_path}")
            return

        # Get total frame count
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.is_open = True
        if self.is_silence is False:
            print(f"Successfully opened video file. Total frames: {self.total_frames}")

        self.img_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.img_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _get_next_frame_from_video(self) -> Optional[cv2.typing.MatLike]:
        """ Read next frame from video. """
        if not self.is_open:
            return None
        
        ret, frame = self.cap.read()
        if ret:
            self.current_index += 1
            return frame
        else:
            # Video reading completed or error occurred
            self.release()
            return None

    # --- Image sequence processing logic ---
    def _init_image_sequence_source(self):
        """ Initialize image sequence folder reading. """
        if not os.path.isdir(self.input_path):
            print(f"Error: Folder not found: {self.input_path}")
            return

        # Find all image files and sort in numerical order
        self.image_files = self._get_sorted_image_files(self.input_path)
        
        if not self.image_files:
            print(f"Error: No image files found in folder {self.input_path}")
            return

        self.total_frames = len(self.image_files)
        self.is_open = True
        if self.is_silence is False:
            print(f"Successfully loaded image sequence. Total images: {self.total_frames}")

        # Read first image to get dimension information
        first_image = cv2.imread(self.image_files[0], cv2.IMREAD_COLOR)
        self.img_height, self.img_width = first_image.shape[:2]

    def _get_next_frame_from_sequence(self) -> Optional[cv2.typing.MatLike]:
        """ Read next image from image sequence. """
        if not self.is_open or self.current_index >= self.total_frames:
            self.release()
            return None

        file_path = self.image_files[self.current_index]
        # cv2.IMREAD_COLOR ensures image is read in color mode
        frame = cv2.imread(file_path, cv2.IMREAD_COLOR)
        
        if frame is None:
             print(f"Warning: Unable to read image file: {file_path}")
        else:
            self.current_index += 1

        return frame

    # --- Core interfaces and helper functions ---
    def get_next_frame(self, device='cpu') -> Optional[cv2.typing.MatLike]:
        """
        [Public interface] Get next image (or frame).

        Returns:
            Optional[cv2.typing.MatLike]: Returns OpenCV image (NumPy array) if successful;
                                       Returns None if end of sequence reached or error occurred.
        """
        if self.is_video:
            color_img =  self._get_next_frame_from_video()
        else:
            color_img =  self._get_next_frame_from_sequence()

        if color_img is None:
            return None, None, False
        else:
            gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            if device != 'cpu':
                gray_img = torch.from_numpy(gray_img).to(device=device).float().unsqueeze(0).unsqueeze(0)     

            return color_img, gray_img, True

    def release(self):
        """ Release resources. """
        if self.is_open:
            if self.is_video and hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            self.is_open = False
        if self.is_silence is False:
            print("\nResources released.")

    def __del__(self):
        """ Ensure resources are released when object is destroyed. """
        self.release()
        
    # --- Natural sorting helper function ---
    @staticmethod
    def _natural_sort_key(s: str) -> List[Union[str, int]]:
        """ Helper function: natural sorting for image sequence. """
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)
        ]

    def _get_sorted_image_files(self, folder_path: str) -> List[str]:
        """ Find and naturally sort image files. """
        extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif')
        all_files = []
        for ext in extensions:
            all_files.extend(glob.glob(os.path.join(folder_path, '*' + ext)))
        
        image_files = [f for f in all_files if os.path.isfile(f)]
        
        # Sort by filename using natural sort
        return sorted(image_files, key=lambda f: self._natural_sort_key(os.path.basename(f)))


class FrameVisualizer:
    def __init__(self, window_name="Visualizer", 
                 result_index_type="matrix",
                 win_width=None, win_height=None, 
                 is_headless=False,
                 conf_threshold=0.8): # 新增阈值参数
        """
        初始化可视化器
        :param conf_threshold: 可视化过滤的相对阈值 (0.0 ~ 1.0)
        """
        self.window_name = window_name
        self.result_index_type = result_index_type  # "matrix", "dots", "bbox"
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
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
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

    def update(self, frame, result, direction=None, process_time=None) -> bool:
        if frame is None:
            return False

        # --- 绘制逻辑 ---
        # 即使 result 是空的，只要不为 None 也可以处理
        if result is not None:
            if self.result_index_type == "matrix":
                self._draw_matrix(frame, result, direction, self.conf_threshold)
            elif self.result_index_type == "dots": 
                self._draw_dots(frame, result, self.conf_threshold)
            elif self.result_index_type == "bbox": 
                self._draw_bbox(frame, result, self.conf_threshold)

        # --- 信息显示 ---
        if process_time is not None:
            cv2.putText(frame, f'Time: {process_time*1000:.1f} ms',
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

            if key == 27: # Esc
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
        if np.max(matrix) <= 0: return

        # np.where 返回 (rows, cols) 即 (y, x)
        rows, cols = np.where(matrix > threshold)
        
        # 画点
        for r, c in zip(rows, cols):
            # cv2 坐标是 (x, y) -> (col, row)
            cv2.drawMarker(frame, (c, r), color=(0, 0, 255), 
                           markerType=cv2.MARKER_STAR, markerSize=5, thickness=1)

        # 画箭头
        if direction_map is not None and len(rows) > 0:
            # 确保 direction_map 维度匹配，这里假设是同样大小的矩阵
            valid_dirs = direction_map[rows, cols]
            
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
    def _draw_bbox(frame, response, threshold):
        """处理 BBox 格式: [[x, y, w, h, score, dir], ...]"""
        if len(response) == 0: return

        scores = response[:, 4]
        
        mask = scores > threshold
        filtered = response[mask]

        if len(filtered) == 0: return

        # 解包前 4 列
        for box in filtered:
            x, y, w, h = box[0:4]
            pt1 = (int(x), int(y))
            pt2 = (int(x + w), int(y + h))
            cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

        # 处理方向 (假设 Col 5 是方向，画在中心)
        if response.shape[1] > 5:
            dirs = filtered[:, 5]
            valid_mask = ~np.isnan(dirs)
            
            valid_boxes = filtered[valid_mask]
            valid_dirs = dirs[valid_mask]
            
            # 计算中心点
            center_xs = valid_boxes[:, 0] + valid_boxes[:, 2] / 2
            center_ys = valid_boxes[:, 1] + valid_boxes[:, 3] / 2
            
            FrameVisualizer._draw_arrows(frame, center_xs, center_ys, valid_dirs)




class ModelSelectorGUI:
    def __init__(self, root):
        self.root = root

    def create_gui(self, modelList):
        self.modelLabel = ttk.Label(self.root, text="Select a model:", width = 15)
        self.modelLabel.grid(row=0, column=0, padx=10, pady=10)

        self.modelCombobox = ttk.Combobox(self.root, values=modelList, width = 30)
        self.modelCombobox.current(11)
        self.modelCombobox.grid(row=0, column=1, columnspan=2, padx=10, pady=10)
        

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
        self.inputTypeLabel = ttk.Label(self.root, text="Select input from:", width = 15)
        self.inputTypeLabel.grid(row=1, column=0, padx=10, pady=10)

        self.selectedOption = tk.IntVar(value=0)


        self.vidLabel = ttk.Radiobutton(self.root, 
                                        text='Video stream', 
                                        variable=self.selectedOption,
                                        value=1, 
                                        command=self.select_vidstream)
        self.vidLabel.grid(row=1, column=2, padx=10, pady=10)
        
        self.imgLabel = ttk.Radiobutton(self.root, 
                                        text='Image stream', 
                                        variable=self.selectedOption,
                                        value=2, 
                                        command=self.select_imgstream)
        self.imgLabel.grid(row=1, column=1, padx=10, pady=10)

    def select_vidstream(self):
        self.imgSelectFolder = None
        self.startImgName = None
        self.endImgName = None
        for element in self.imgElement.values():
            element.destroy()

        self.vidElement['lblVidIndicate'] = ttk.Label(self.root, text= 'Video\'s path:',  width = 15)
        self.vidElement['lblVidIndicate'].grid(row=2, column=0, padx=10, pady=30)
        self.vidElement['lblVidPath'] = ttk.Label(self.root, 
                                           text="Waiting for the selection",
                                           wraplength=220
                                           )
        self.vidElement['lblVidPath'].grid(row=2, column=1, columnspan=2, padx=10, pady=10)
        
        self.vidElement['btn'] = ttk.Button(self.root, text="Select a video", command=self._clicked_vid)
        self.vidElement['btn'].grid(row=3, column=2, padx=10, pady=10)
        
    def _clicked_vid(self):
        self.vidName = filedialog.askopenfilenames(initialdir=VID_DEFAULT_FOLDER)
        self.vidName = self.vidName[0]
        self.vidElement['lblVidPath'].config(text=self.vidName)

    def select_imgstream(self):
        self.vidName = None
        for element in self.vidElement.values():
            element.destroy()

        self.imgElement['lblFolder'] = ttk.Label(self.root, text="Image's folder: ",  width = 15)
        self.imgElement['lblFolder'].grid(row=2, column=0, padx=10, pady=10)
        self.imgElement['lblFolderName'] = ttk.Label(self.root, text="Waiting for the selection", wraplength=220)
        self.imgElement['lblFolderName'].grid(row=2, column=1, columnspan=2, padx=10, pady=30)

        self.imgElement['btnStart'] = ttk.Button(self.root, text="Select start frame", command=self._clicked_start_img)
        self.imgElement['btnStart'].grid(row=3, column=1,  padx=10, pady=10)
        self.imgElement['btnEnd'] = ttk.Button(self.root, text="Select end frame", command=self._clicked_end_img)
        self.imgElement['btnEnd'].grid(row=4, column=1,  padx=10, pady=10)
        
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

        self.imgElement['lblStartImg'] = ttk.Label(self.root, text=self.startImgName)
        self.imgElement['lblStartImg'].grid(row=3, column=2, padx=10, pady=10)

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

        self.imgElement['lblEndImg'] = ttk.Label(self.root, text=self.endImgName)
        self.imgElement['lblEndImg'].grid(row=4, column=2, padx=10, pady=10)


class ModelAndInputSelectorGUI:
    def __init__(self, root):
        self.root = root

        windowHeight = 350
        windowWidth = 400
        
        startHeight = (root.winfo_screenheight() - windowHeight) // 2
        startWidth = (root.winfo_screenwidth() - windowWidth) // 2

        self.root.geometry('{}x{}+{}+{}'.format(windowWidth, windowHeight, startWidth, startHeight))
        self.root.title("Small target motion detector - Runner")
        self.root.iconbitmap(os.path.join(os.path.dirname(filePath), 'stmd.ico'))
        
        self.objModelSelector = ModelSelectorGUI(root)
        self.objInputSelector = InputSelectorGUI(root)
        
        self.btnRun = ttk.Button(self.root, text="Run", command=self._run)
        self.btnRun.place(x = 20, y=300)
        self.btnStepping = ttk.Button(self.root, text="Stepping", command=self._stepping)
        self.btnStepping.place(x = 20, y=270)
        self.isStepping = False

    def create_gui(self):
        self.objModelSelector.create_gui(ALL_MODEL)
        self.objInputSelector.create_gui()

        self.root.mainloop()

        if self.objInputSelector.selectedOption.get() == 1:  
            return self.modelName, self.vidName, None, self.isStepping
        elif self.objInputSelector.selectedOption.get() == 2:
            return self.modelName, self.startImgName, self.endImgName, self.isStepping

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


def check_same_ext_name(startImgName, endImgName):
    _, ext1 = os.path.splitext(startImgName)
    _, ext2 = os.path.splitext(endImgName)
    # Check if the extensions of the start and end images are the same
    if ext1 != ext2:
        return False
    else:
        return True
    

