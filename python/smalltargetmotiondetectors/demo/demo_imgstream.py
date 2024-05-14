# demo_imgstearm
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

''' Create an image stream reader '''
objIptStream = ImgstreamReader(os.path.join(filePath[:indexPath-7], 'demodata', 'imgstream', 'DemoFig*.jpg'))

''' Alternatively, uncomment the following options for different inputs: '''
# objIptStream = ImgstreamReader(os.path.join(filePath[:indexPath-7], 'demodata', 'imgstream', 'DemoFig*.jpg'), 10,)
# objIptStream = ImgstreamReader(os.path.join(filePath[:indexPath-7], 'demodata', 'imgstream', 'DemoFig*.jpg'), 10, 100 )

''' Get visualization handle and initiate model '''
# Get visualization handle
objVisualize = get_visualize_handle(objModel.__class__.__name__)
# Initialize the model
objModel.init_config()

''' Run '''
while objIptStream.hasFrame and objVisualize.hasFigHandle:
    # Read the next frame from the image stream
    grayImg, colorImg = objIptStream.get_next_frame()
    
    # Perform inference using the objModel
    result = inference(objModel, grayImg)
    
    # Display the result
    objVisualize.show_result(colorImg, result)
