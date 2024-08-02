# demo_vidstream
import os
import sys

# Get the full path of this file
filePath = os.path.realpath(__file__)
# Find the index of '/+smalltargetmotiondetectors/'
# in the file path
indexPath = filePath.rfind(os.path.sep + 'smalltargetmotiondetectors' + os.path.sep)
# Add the path to the package containing the models
sys.path.append(filePath[:indexPath])

from smalltargetmotiondetectors.api import *
from smalltargetmotiondetectors.util.iostream import *
from smalltargetmotiondetectors.model import *

''' Instantiate the model '''
objModel = instancing_model('Backbonev2')

''' Create a video stream reader '''
objIptStream = VidstreamReader(os.path.join(filePath[:indexPath-7], 'demodata', 'RIST_GX010290_orignal_240Hz.mp4'))

''' Alternatively, uncomment the following options for different inputs: '''
# objIptStream = VidstreamReader(os.path.join(filePath[:indexPath-7], 'demodata', 'RIST_GX010290_orignal_240Hz.mp4'), 1)
# objIptStream = VidstreamReader(os.path.join(filePath[:indexPath-7], 'demodata', 'RIST_GX010290_orignal_240Hz.mp4'), 1, 100)
# objIptStream = VidstreamReader(os.path.join(filePath[:indexPath-7], 'demodata', 'simulatedVideo0_orignal_1000Hz.mp4'))

''' Get visualization handle and initiate model '''
# Get visualization handle
objVisualize = get_visualize_handle(objModel.__class__.__name__)

''' Initialize the model '''
# set the parameter list
objModel.set_parameter()
# print the parameter list
objModel.print_parameter()
# init
objModel.init_config()

''' Run '''
while objIptStream.hasFrame and objVisualize.hasFigHandle:
    # Read the next frame from the video stream
    grayImg, colorImg = objIptStream.get_next_frame()
    
    # Perform inference using the model
    result = inference(objModel, grayImg)
    
    # Display the result
    objVisualize.show_result(colorImg, result)
