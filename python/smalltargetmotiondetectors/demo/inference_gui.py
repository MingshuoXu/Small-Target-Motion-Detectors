import os
import sys
import queue
import signal
import logging
from typing import Optional, Tuple
from multiprocessing import Process, Queue, Event
import tkinter as tk

# configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StmdGui:
    def __init__(self):
        self._setup_paths()
        
    def _setup_paths(self):
        """ setup paths """
        file_path = os.path.realpath(__file__)
        self.py_pkg_path = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
        sys.path.append(self.py_pkg_path)
        
        from smalltargetmotiondetectors.util.iostream import ( # type: ignore
            ModelAndInputSelectorGUI,
            ImgstreamReader,
            VidstreamReader
        )
        from smalltargetmotiondetectors.api import ( # type: ignore
            instancing_model,
            get_visualize_handle,
            inference
        ) 
        
        self.ModelAndInputSelectorGUI = ModelAndInputSelectorGUI
        self.ImgstreamReader = ImgstreamReader
        self.VidstreamReader = VidstreamReader
        self.instancing_model = instancing_model
        self.get_visualize_handle = get_visualize_handle
        self.inference = inference

    def _get_user_input(self) -> tuple:
        """ get user input """
        root = tk.Tk()
        gui = self.ModelAndInputSelectorGUI(root)
        return gui.create_gui()

    def _create_queues(self) -> Tuple[Queue, Queue]:
        """ create queues """
        return Queue(maxsize=3), Queue(maxsize=3)

    def _start_processes(self, ipt_queue: Queue, res_queue: Queue, 
                        model_name: str, opt1: str, opt2: Optional[str], 
                        is_stepping: bool) -> Tuple:
        """ start processes """
        exit_event = Event()
        processes = [
            Process(target=self._read_frames, args=(ipt_queue, opt1, opt2, exit_event), name="FrameReader"),
            Process(target=self._run_inference, args=(ipt_queue, res_queue, model_name, exit_event), name="Inference"),
            Process(target=self._visualize_results, args=(res_queue, model_name, is_stepping, exit_event), name="Visualizer")
        ]
        
        for p in processes:
            p.daemon = True  # set as daemon process
            p.start()
            
        return exit_event, processes

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
            model = self.instancing_model(model_name)
            model.init_config()
            
            while not exit_event.is_set():
                if (data := self._safe_get(ipt_queue, exit_event)) is None:
                    break
                    
                result = self.inference(model, data[0])
                self._safe_put(res_queue, (data[1], result), exit_event)
                
            self._safe_put(res_queue, None, exit_event)  # terminate signal
            logger.info("Inference process exited cleanly")
            
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            exit_event.set()

    def _visualize_results(self, res_queue: Queue, model_name: str, is_stepping: bool, exit_event):
        """ visualization process """
        try:
            visualizer = self.get_visualize_handle(model_name)
            if is_stepping:
                visualizer.uiHandle["pauseButton"].bool = True
                visualizer._pauseCallback()

            while not exit_event.is_set():
                if not visualizer.hasFigHandle:
                    break
                    
                if (data := self._safe_get(res_queue, exit_event)) is None:
                    break
                    
                visualizer.show_result(*data)
                
            if visualizer.hasFigHandle:
                del(visualizer)
            exit_event.set()
            logger.info("Visualization exited cleanly")
            
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")
            exit_event.set()

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
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def run(self):
        """ run video processor"""
        # handle keyboard interrupt
        original_sigint = signal.signal(signal.SIGINT, lambda *_: exit_event.set())
        
        try:
            # get user input
            model_name, opt1, opt2, is_stepping = self._get_user_input()
            
            # create queues
            ipt_queue, res_queue = self._create_queues()
            
            # start processes
            exit_event, processes = self._start_processes(ipt_queue, res_queue, 
                                                         model_name, opt1, opt2, 
                                                         is_stepping)
            
            # wait for processes to finish
            while not exit_event.is_set():
                if not any(p.is_alive() for p in processes):
                    break
                exit_event.wait(0.5) 
                
        except Exception as e:
            logger.error(f"Main process error: {str(e)}")
        finally:
            # clean up resources
            exit_event.set()
            
            # wait for processes to finish
            for p in processes:
                if p.is_alive():
                    p.join(timeout=0.01)
                if p.exitcode is None:
                    p.terminate()
                    
            # clear queues
            self._cleanup_queues(ipt_queue, res_queue)
            
            # restore original signal handler
            signal.signal(signal.SIGINT, original_sigint)
            logger.info("Shutdown completed")

if __name__ == "__main__":
    obj = StmdGui()
    obj.run()