import os
import sys
import queue

from multiprocessing import Process, Queue, Event
import tkinter as tk
import time


filePath = os.path.realpath(__file__)
pyPackagePath = os.path.dirname(os.path.dirname(os.path.dirname(filePath)))
gitCodePath = os.path.dirname(pyPackagePath)
sys.path.append(pyPackagePath)

from smalltargetmotiondetectors.util.iostream import (ModelAndInputSelectorGUI, # type: ignore
                                                      ImgstreamReader,
                                                      VidstreamReader,
                                                      )  # type: ignore
from smalltargetmotiondetectors.api import instancing_model, get_visualize_handle, inference  # type: ignore


''' ref: https://zhuanlan.zhihu.com/p/675873139 '''

def read_frames(iptQ, opt1, opt2, exit_event):
    """ Initialize input stream based on user options."""
    if opt2 is not None:
        objIptStream = ImgstreamReader(startImgName=opt1, endImgName=opt2)
    else:
        objIptStream = VidstreamReader(vidName=opt1)
        
    # Read frames and put them into the input queue
    while not exit_event.is_set() and objIptStream.hasFrame:
        grayImg, colorImg = objIptStream.get_next_frame()
        
        # Put the frame into the input queue
        while not exit_event.is_set():
            try:
                iptQ.put((grayImg, colorImg), timeout=0.1)
                break
            except queue.Full:
                continue

    # Put None to signal the end of the input stream
    while not exit_event.is_set():
        try:
            iptQ.put(None, timeout=0.1)
            break
        except queue.Full:
            continue
    exit_event.set()


def run_inference(iptQ, resQ, modelName, exit_event):
    """Run inference and put results into the result queue."""
    objModel = instancing_model(modelName)
    objModel.init_config()

    while not exit_event.is_set():
        try:
            data = iptQ.get(timeout=0.1)
            if data is None:  # End of input stream
                while not exit_event.is_set():
                    try:
                        resQ.put(None, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                exit_event.set()
                break
                
            grayImg, colorImg = data
            result = inference(objModel, grayImg)
            
            # Put the result into the result queue
            while not exit_event.is_set():
                try:
                    resQ.put((colorImg, result), timeout=0.1)
                    break
                except queue.Full:
                    continue
                    
        except queue.Empty:
            continue
            
    exit_event.set()


def visualize_results(resQ, modelName, isStepping, exit_event):
    """Visualize results."""
    objVisualize = get_visualize_handle(modelName)
    if isStepping:
        objVisualize.uiHandle["pauseButton"].bool = True
        objVisualize._pauseCallback()

    while not exit_event.is_set():
        # Check if the visualization window is closed
        if not objVisualize.hasFigHandle:
            exit_event.set()
            break
            
        try:
            data = resQ.get(timeout=0.1)
            if data is None:    # End of input stream
                exit_event.set()
                break
                
            colorImg, result = data
            objVisualize.show_result(colorImg, result)
            
        except queue.Empty:
            continue
            
    # Close the visualization window
    if objVisualize.hasFigHandle:
        del(objVisualize)
    exit_event.set()


if __name__ == "__main__":
    # Open GUI to get user input
    root = tk.Tk()
    objGUI = ModelAndInputSelectorGUI(root)
    modelName, opt1, opt2, isStepping = objGUI.create_gui()

    # Create queues
    iptQueue = Queue(maxsize=3)
    resQueue = Queue(maxsize=3)
    exit_event = Event()

    # Create and start processes
    processes = [
        Process(target=read_frames, args=(iptQueue, opt1, opt2, exit_event)),
        Process(target=run_inference, args=(iptQueue, resQueue, modelName, exit_event)),
        Process(target=visualize_results, args=(resQueue, modelName, isStepping, exit_event))
    ]
    for p in processes:
        p.start()

    # mian process monitor
    try:
        while any(p.is_alive() for p in processes):
            time.sleep(0.1)
            if exit_event.is_set():
                break
                
    except KeyboardInterrupt:
        exit_event.set()
        print("\n User interrupt, terminating processes...")
        
    finally:
        # Terminate all processes
        for p in processes:
            p.join(timeout=1)
            if p.is_alive():
                p.terminate()
                
        # Clear queues
        while not iptQueue.empty():
            iptQueue.get()
        while not resQueue.empty():
            resQueue.get()

        print("All processes terminated.")

