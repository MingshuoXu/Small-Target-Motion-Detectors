import os
import sys
import time

# Get the full path of this file
filePath = os.path.realpath(__file__)
# Find the index of '/+smalltargetmotiondetectors/'
indexPath = filePath.rfind('smalltargetmotiondetectors')
# Add the path to the package containing the models 
sys.path.append(filePath[:indexPath])

from smalltargetmotiondetectors.api import *
from smalltargetmotiondetectors.util.iostream import *

''' Model instantiation '''
objModel = instancing_model('HaarSTMD')

''' Input '''
hSteam = VidstreamReader(os.path.join(filePath[:indexPath-7], 'demodata', 'simulatedVideo0_orignal_1000Hz.mp4'))
# hSteam = VidstreamReader(os.path.join(filePath[:indexPath-7], 'demodata', 'RIST_GX010290_orignal_240Hz.mp4'))

''' Get visualization handle '''
hVisual = get_visualize_handle(objModel.__class__.__name__)

''' Initialize the model '''
# set the parameter list
objModel.set_parameter( sigma1  = 1,
                        n1      = 10,
                        tau1    = 3,
                        n2      = 10,
                        tau2    = 9,
                        sigma2  = 1.5,
                        sigma3  = 3,
                        TAU     = 1,
                        )
# print the parameter list
objModel.print_parameter()
# init
objModel.init_config()


''' Run inference '''
while hSteam.hasFrame and hVisual.hasFigHandle:

    # Get the next frame from the input source
    grayImg, colorImg = hSteam.get_next_frame()
    
    # Perform inference using the model
    result = inference(objModel, grayImg)
    
    # Visualize the result
    hVisual.show_result(colorImg, result)



