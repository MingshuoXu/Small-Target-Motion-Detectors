import tkinter as tk

from python.smalltargetmotiondetectors.util.iostream import ModelAndInputSelectorGUI, ImgstreamReader, VidstreamReader
from python.smalltargetmotiondetectors.api import instancing_model, get_visualize_handle, inference
from python.smalltargetmotiondetectors.model import *

'''
    main
'''

root = tk.Tk()

obj = ModelAndInputSelectorGUI(root)
modelName, opt1, opt2 = obj.create_gui()

objModel = instancing_model(modelName)
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
