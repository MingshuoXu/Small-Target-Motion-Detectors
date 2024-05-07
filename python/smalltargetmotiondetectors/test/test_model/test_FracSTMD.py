import os
import cv2
import sys

# Get the full path of this file
filePath = os.path.realpath(__file__)
# Find the index of '/+smalltargetmotiondetectors/'
indexPath = filePath.rfind('smalltargetmotiondetectors')
# Add the path to the package containing the models
import_path = filePath[:indexPath]
sys.path.append(import_path)

from smalltargetmotiondetectors.api import *
from smalltargetmotiondetectors.util.iostream import *

# Model instantiation
objModel = instancing_model('FracSTMD')

# Input
# Demo video (RIST)
hSteam = VidstreamReader(os.path.join(filePath[:indexPath-7], 'demodata', 'RIST_GX010290.mp4'))

# Visualization and model init
# Get visualization handle
hVisual = get_visualize_handle(objModel.__class__.__name__)

# Initialize the model
objModel.init_config()

# Run inference
while hSteam.hasFrame and hVisual.hasFigHandle:
    # Get the next frame from the input source
    grayImg, colorImg = hSteam.get_next_frame()
    
    # Perform inference using the model
    result = inference(objModel, grayImg)
    
    # Visualize the result
    hVisual.show_result(colorImg, result)
