# demo_vidstream
import os
import sys

# Get the full path of this file
filePath = os.path.realpath(__file__)
# Find the index of '/+smalltargetmotiondetectors/'
# in the file path
indexPath = filePath.rfind(os.path.sep + '+smalltargetmotiondetectors' + os.path.sep)
# Add the path to the package containing the models
sys.path.append(filePath[:indexPath])

# Import necessary modules
from smalltargetmotiondetectors.api import *
from smalltargetmotiondetectors.util.iostream import *

# Instantiate the model
model = instancing_model()

# Create a video stream reader
# Specify the path to the video file or uncomment one of the provided options
hSteam = VidstreamReader()

# Alternatively, uncomment the following options for different inputs:

# Demo video (RIST)
# hSteam = VidstreamReader( ...
#     filePath[:indexPath-7] + 'demodata' + os.path.sep + 'RIST_GX010290.mp4')

# Get visualization handle and initiate model

# Get visualization handle
hVisual = get_visualize_handle(model.__class__.__name__, 0.5)

# Initialize the model
model.init_config()

# Run
while hSteam.hasFrame and hVisual.hasFigHandle:
    # Read the next frame from the video stream
    grayImg, colorImg = hSteam.get_next_frame()
    
    # Perform inference using the model
    result = inference(model, grayImg)
    
    # Display the result
    hVisual.show_result(colorImg, result)
