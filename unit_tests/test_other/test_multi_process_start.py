import os
import sys
import time

from multiprocessing import Process, Queue, Event
import tkinter as tk
import time


# Add the package path to the system path
filePath = os.path.realpath(__file__)
packagePath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(filePath))))
sys.path.append(packagePath)

from smalltargetmotiondetectors.demo.inference_gui import StmdGui # type: ignore
from smalltargetmotiondetectors.api import instancing_model, get_visualize_handle, inference # type: ignore
from smalltargetmotiondetectors.util.iostream import VidstreamReader # type: ignore

def task1():
    obj = StmdGui()
        
    try:
        # get user input
        modelName = 'STMDNet'
        opt1 = os.path.join(os.path.dirname(packagePath),
                            'demodata', 'RIST_GX010290_orignal_240Hz.mp4')
        opt2 = None
        isStepping = False
        
        # create queues
        ipt_queue, res_queue = obj._create_queues()
        
        # start processes
        timetic = time.time()
        exit_event, processes = obj._start_processes(ipt_queue, res_queue, 
                                                        modelName, opt1, opt2, 
                                                        isStepping)
        
        # wait for processes to finish
        while not exit_event.is_set():
            if not any(p.is_alive() for p in processes):
                break
            exit_event.wait(0.5) 
            
    except Exception as e:
        print("An error occurred: ", e)
    finally:
        # clean up resources
        timetoc = time.time() - timetic
        print('Multi process\n\tTime taken: {:.2f} seconds'.format(timetoc))

        exit_event.set()
        
        # wait for processes to finish
        for p in processes:
            if p.is_alive():
                p.join(timeout=2)
            if p.exitcode is None:
                p.terminate()
                
        # clear queues
        obj._cleanup_queues(ipt_queue, res_queue)
    



def task2():
    modelName = 'STMDNet'
    opt1 = os.path.join(os.path.dirname(packagePath),
                        'demodata', 'RIST_GX010290_orignal_240Hz.mp4')
    isStepping = False

    objModel = instancing_model(modelName)
    objIptStream = VidstreamReader(vidName=opt1)

    ''' Get visualization handle '''
    objVisualize = get_visualize_handle(objModel.__class__.__name__)
    if isStepping:
        objVisualize.uiHandle['pauseButton'].bool = True
        objVisualize._pauseCallback()

    ''' Initialize the model '''
    objModel.init_config()

    timetic = time.time()
    ''' Run inference '''
    while objIptStream.hasFrame and objVisualize.hasFigHandle:
        # Get the next frame from the input source
        grayImg, colorImg = objIptStream.get_next_frame()
        
        # Perform inference using the model
        result = inference(objModel, grayImg)
        
        # Visualize the result
        objVisualize.show_result(colorImg, result)
    timetoc = time.time() - timetic
    print('Single process\n\tTime taken: {:.2f} seconds'.format(timetoc))

if __name__ == '__main__':
    task1()
    task2()
    
