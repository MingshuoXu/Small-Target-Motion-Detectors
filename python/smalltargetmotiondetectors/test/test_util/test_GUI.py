import os
import sys
import cv2
import tkinter as tk

# Get the full path of this file
filePath = os.path.abspath(__file__)
# Find the index of '/smalltargetmotiondetectors/' in the file path
indexPath = filePath.find(os.path.join(os.sep, 'smalltargetmotiondetectors'))
# Add the path to the package containing the models
sys.path.append(filePath[:indexPath])

from smalltargetmotiondetectors.util.iostream import *
from smalltargetmotiondetectors.api import *
from smalltargetmotiondetectors.model import *

if __name__ == "__main__":

    root = tk.Tk()
 
    
    obj = ModelAndInputSelectorGUI(root)
    modelName, opt1, opt2 = obj.create_gui()

    objModel = eval(modelName)
    if opt2:
        objIptStream = ImgstreamReader(startImgName=opt1, endImgName=opt2)
    else:
        objIptStream = VidstreamReader(vidName=opt1)

    # Get visualization handle
    objVisual = get_visualize_handle(objModel.__class__.__name__)

    # Initialize the model
    objModel.init_config()

    # Run inference
    while objIptStream.hasFrame and objVisual.hasFigHandle:
        # Get the next frame from the input source
        grayImg, colorImg = objIptStream.get_next_frame()
        
        # Perform inference using the model
        result = inference(objModel, grayImg)
        
        # Visualize the result
        objVisual.show_result(colorImg, result)
    