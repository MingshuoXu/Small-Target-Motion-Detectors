import os
import sys
import queue
import signal
import logging
from typing import Optional, Tuple
from multiprocessing import Process, Queue, Event
import time


import tkinter as tk
import torch

file_path = os.path.realpath(__file__)
py_pkg_path = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
sys.path.append(py_pkg_path)

try:
    from xttmp.util.iostream import ( # type: ignore
                    ModelAndInputSelectorGUI,
                    ImgstreamReader,
                    VidstreamReader
                )
    from xttmp.api import ( # type: ignore
        instancing_model,
        get_visualize_handle,
    ) 
except ImportError as e:
    raise ImportError("Failed to import required modules. "
                      "Ensure that 'smalltargetmotiondetectors' package is correctly installed.") from e

# configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StmdGui:
    def __init__(self, device='cpu', show_threshold: float = 0.8):
        """ Initialize STMD GUI """
        self.device = device
        self.show_threshold = show_threshold
        self._setup_paths()
        
    def _setup_paths(self):
        self.ModelAndInputSelectorGUI = ModelAndInputSelectorGUI
        self.ImgstreamReader = ImgstreamReader
        self.VidstreamReader = VidstreamReader
        self.instancing_model = instancing_model
        self.get_visualize_handle = get_visualize_handle


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

    def _create_queues(self) -> Tuple[Queue, Queue]:
        """ create queues """
        return Queue(maxsize=3), Queue(maxsize=3)

    def _start_processes(self, ipt_queue: Queue, res_queue: Queue, 
                        model_name: str, opt1: str, opt2: Optional[str], 
                        is_stepping: bool, exit_event) -> list:
        """ start processes """
        # FIX: 移除内部创建 exit_event，改为使用传入的 event，确保主进程信号能传递给子进程
        
        processes = [
            Process(target=self._read_frames, args=(ipt_queue, opt1, opt2, exit_event), name="FrameReader"),
            Process(target=self._run_inference, args=(ipt_queue, res_queue, model_name, exit_event), name="Inference"),
            Process(target=self._visualize_results, args=(res_queue, model_name, is_stepping, exit_event, self.show_threshold), name="Visualizer")
        ]
        
        for p in processes:
            p.daemon = True  # set as daemon process
            p.start()
            
        return processes

    def _read_frames(self, ipt_queue: Queue, opt1: str, opt2: Optional[str], exit_event):
        """ frame reader process """
        try:
            reader = self.ImgstreamReader(startImgName=opt1, endImgName=opt2) if opt2 else self.VidstreamReader(vidName=opt1)
            
            while not exit_event.is_set() and reader.hasFrame:
                frame_data = reader.get_next_frame()
                self._safe_put(ipt_queue, frame_data, exit_event)
                
            self._safe_put(ipt_queue, None, exit_event)  # terminate signal
            logger.info("Frame reader exited cleanly")
            
        except Exception as e:
            logger.error(f"Frame reader failed: {str(e)}")
            exit_event.set()

    def _run_inference(self, ipt_queue: Queue, res_queue: Queue, model_name: str, exit_event):
        """ inference process """
        try:
            model = self.instancing_model(model_name, device = self.device)
            model.init_config()
            
            while not exit_event.is_set():
                if (data := self._safe_get(ipt_queue, exit_event)) is None:
                    break
                
                # Input handling
                if self.device != 'cpu':
                    # Ensure numpy array is converted properly before unsqueeze
                    _ipt = torch.from_numpy(data[0]).float().to(self.device).unsqueeze(0).unsqueeze(0)
                else:
                    _ipt = data[0]

                result, runTime = model.process(_ipt)

                # Output handling
                if self.device != 'cpu':
                    # Detach is safer before numpy conversion
                    result = {k: v.detach().cpu().numpy().squeeze(0).squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in result.items()}
                
                self._safe_put(res_queue, (data[1], result, runTime), exit_event)
                
            self._safe_put(res_queue, None, exit_event)  # terminate signal
            logger.info("Inference process exited cleanly")
            
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            exit_event.set()

    def _visualize_results(self, res_queue: Queue, model_name: str, is_stepping: bool, exit_event, show_threshold: float = 0.8):
        """ visualization process """
        visualizer = None
        try:
            visualizer = self.get_visualize_handle(model_name, show_threshold)
            if is_stepping:
                visualizer.uiHandle["pauseButton"].bool = True
                visualizer._pauseCallback()

            while not exit_event.is_set():
                if not visualizer.hasFigHandle:
                    break
                    
                if (data := self._safe_get(res_queue, exit_event)) is None:
                    break
                    
                visualizer.show_result(*data)
            
            # 正常退出循环也应该设置 Event，通知其他进程
            exit_event.set() 
            logger.info("Visualization exited cleanly")
            
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")
            exit_event.set()
        finally:
            # FIX: 安全的资源清理
            if visualizer and hasattr(visualizer, 'hasFigHandle') and visualizer.hasFigHandle:
                del visualizer

    def _safe_put(self, q: Queue, item, exit_event, timeout: float = 0.1):
        """ safe queue put """
        while not exit_event.is_set():
            try:
                q.put(item, timeout=timeout)
                return
            except queue.Full:
                continue

    def _safe_get(self, q: Queue, exit_event, timeout: float = 0.1):
        """ safe queue get """
        while not exit_event.is_set():
            try:
                return q.get(timeout=timeout)
            except queue.Empty:
                continue
        return None
    
    def _cleanup_queues(self, *queues):
        """ clear queues """
        for q in queues:
            if q is None: continue
            try:
                while not q.empty():
                    q.get_nowait()
            except (queue.Empty, OSError, ValueError):
                # OSError/ValueError 处理队列已关闭的情况
                pass

    def run(self):
        """ run video processor"""
        # FIX: 在 try 块外部初始化关键变量，防止 finally 块报 UnboundLocalError
        exit_event = Event()
        processes = []
        ipt_queue = None
        res_queue = None

        # FIX: Signal Handler 必须在 exit_event 定义之后
        def signal_handler(*args):
            logger.info("Interrupt received, stopping...")
            exit_event.set()
            
        original_sigint = signal.signal(signal.SIGINT, signal_handler)
        
        try:
            # get user input
            # 如果这里抛出异常（比如用户直接关闭窗口），程序会进入 finally
            user_input = self._get_user_input()
            if not user_input: # 处理用户取消输入的情况
                logger.info("User cancelled input.")
                return 

            model_name, opt1, opt2, is_stepping = user_input
            
            # create queues
            ipt_queue, res_queue = self._create_queues()
            
            # start processes (Passing the MAIN exit_event)
            processes = self._start_processes(ipt_queue, res_queue, 
                                            model_name, opt1, opt2, 
                                            is_stepping, exit_event)
            
            # wait for processes to finish
            while not exit_event.is_set():
                if not any(p.is_alive() for p in processes):
                    break
                exit_event.wait(0.5) 
                
        except Exception as e:
            logger.error(f"Main process error: {str(e)}")
        finally:
            logger.info("Cleaning up resources...")
            exit_event.set() # 1. 先通知子进程停止生产
            
            # FIX: 【关键修改】先清空队列，再 Join 进程
            # 必须先把队列里的残留数据读出来，子进程的后台线程才能释放，进程才能真正退出
            if ipt_queue or res_queue:
                logger.info("Draining queues to unblock processes...")
                # 循环多次确保读空，防止正好有数据刚写进去
                for _ in range(10): 
                    self._cleanup_queues(ipt_queue, res_queue)
                    time.sleep(0.05) 
                    if (ipt_queue is None or ipt_queue.empty()) and \
                       (res_queue is None or res_queue.empty()):
                        break

            # 3. 现在可以安全地等待进程结束了
            if processes:
                for p in processes:
                    if p.is_alive():
                        # 因为队列已清空，这里通常能瞬间 join 成功
                        p.join(timeout=1.0) 
                    if p.is_alive():
                        logger.warning(f"Process {p.name} did not exit, terminating...")
                        p.terminate()
            
            # double check cleanup
            if ipt_queue and res_queue:
                self._cleanup_queues(ipt_queue, res_queue)
                # 显式关闭队列资源
                ipt_queue.close()
                res_queue.close()
                # join_thread 并非必须，但在某些系统上能防止 broken pipe
                # ipt_queue.join_thread() 
            
            signal.signal(signal.SIGINT, original_sigint)
            logger.info("Shutdown completed")

def main(show_threshold: float = 0.8):
    # 增加 multiprocessing start method 保护 (特别是 Windows/MacOS)
    try:
        from multiprocessing import set_start_method
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DEVICE = 'cpu'
    obj = StmdGui(DEVICE, show_threshold)
    obj.run()

if __name__ == "__main__":
    main(0.2)