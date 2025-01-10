# demo_vidstream
import os
import sys

filePath = os.path.realpath(__file__)
pyPackagePath = os.path.dirname(os.path.dirname(os.path.dirname(filePath)))
gitCodePath = os.path.dirname(pyPackagePath)
sys.path.append(pyPackagePath)

from smalltargetmotiondetectors.api import *
from smalltargetmotiondetectors.util.iostream import *
from smalltargetmotiondetectors.model import *

''' Instantiate the model '''
objModel = instancing_model('Backbonev2')

''' Create a video stream reader '''
objIptStream = VidstreamReader(os.path.join(gitCodePath, 'demodata', 'RIST_GX010290_orignal_240Hz.mp4'))

''' Alternatively, uncomment the following options for different inputs: '''
# objIptStream = VidstreamReader(os.path.join(gitCodePath, 'demodata', 'RIST_GX010290_orignal_240Hz.mp4'), 1)
# objIptStream = VidstreamReader(os.path.join(gitCodePath, 'demodata', 'RIST_GX010290_orignal_240Hz.mp4'), 1, 100)
# objIptStream = VidstreamReader(os.path.join(gitCodePath, 'demodata', 'simulatedVideo0_orignal_1000Hz.mp4'))

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
