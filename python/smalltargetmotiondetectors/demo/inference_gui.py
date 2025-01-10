import os
import sys

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
    """Initialize input stream based on user options."""
    if opt2 is not None:
        objIptStream = ImgstreamReader(startImgName=opt1, endImgName=opt2)
    else:
        objIptStream = VidstreamReader(vidName=opt1)
    """Read frames and put them into the input queue."""
    while objIptStream.hasFrame and not exit_event.is_set():
        grayImg, colorImg = objIptStream.get_next_frame()
        iptQ.put((grayImg, colorImg))
    iptQ.put(None)  # End signal
    exit_event.set()


def run_inference(iptQ, resQ, modelName, exit_event):
    """Run inference and put results into the result queue."""
    objModel = instancing_model(modelName)
    objModel.init_config()
    while not exit_event.is_set():
        data = iptQ.get()
        if data is None:
            resQ.put(None)  # End signal
            exit_event.set()
            break
        grayImg, colorImg = data
        result = inference(objModel, grayImg)
        resQ.put((colorImg, result))


def visualize_results(resQ, modelName, isStepping, exit_event):
    """Visualize results."""
    objVisualize = get_visualize_handle(modelName)
    if isStepping:
        objVisualize.uiHandle["pauseButton"].bool = True
        objVisualize._pauseCallback()

    while not exit_event.is_set():
        if not objVisualize.hasFigHandle:
            exit_event.set()
            break

        data = resQ.get()
        if data is None:
            exit_event.set()
            break
        else:
            colorImg, result = data
        
        objVisualize.show_result(colorImg, result)
        
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
    process1 = Process(target=read_frames, args=(iptQueue, opt1, opt2, exit_event))
    process2 = Process(target=run_inference, args=(iptQueue, resQueue, modelName, exit_event))
    process3 = Process(target=visualize_results, args=(resQueue, modelName, isStepping, exit_event))

    process1.start()
    process2.start()
    process3.start()

    process1.join()
    process2.join()
    process3.join()
