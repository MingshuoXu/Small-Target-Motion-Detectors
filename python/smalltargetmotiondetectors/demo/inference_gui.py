import os
import sys
import tkinter as tk

# Get the full path of this file
filePath = os.path.abspath(__file__)
# Find the index of '/smalltargetmotiondetectors/' in the file path
indexPath = filePath.find(os.path.join(os.sep, 'smalltargetmotiondetectors'))
# Add the path to the package containing the models
sys.path.append(filePath[:indexPath])

from smalltargetmotiondetectors.util.iostream import ModelAndInputSelectorGUI, ImgstreamReader, VidstreamReader
from smalltargetmotiondetectors.api import instancing_model, get_visualize_handle, inference
from smalltargetmotiondetectors.model import *

''' open gui to get modelName and type of inputstream '''
root = tk.Tk()
obj = ModelAndInputSelectorGUI(root)
modelName, opt1, opt2 = obj.create_gui()

objModel = instancing_model(modelName)
if opt2:
    objIptStream = ImgstreamReader(startImgName=opt1, endImgName=opt2)
else:
    objIptStream = VidstreamReader(vidName=opt1)

''' Get visualization handle '''
objVisualize = get_visualize_handle(objModel.__class__.__name__)

''' Initialize the model '''
objModel.init_config()

''' Run inference '''
while objIptStream.hasFrame and objVisualize.hasFigHandle:
    # Get the next frame from the input source
    grayImg, colorImg = objIptStream.get_next_frame()
    
    # Perform inference using the model
    result = inference(objModel, grayImg)
    
    # Visualize the result
    objVisualize.show_result(colorImg, result)
