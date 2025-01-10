# demo_imgstearm
import os
import sys

filePath = os.path.realpath(__file__)
pyPackagePath = os.path.dirname(os.path.dirname(os.path.dirname(filePath)))
gitCodePath = os.path.dirname(pyPackagePath)
sys.path.append(pyPackagePath)

from smalltargetmotiondetectors.api import instancing_model, get_visualize_handle, inference # type: ignore
from smalltargetmotiondetectors.util.iostream import ImgstreamReader # type: ignore
from smalltargetmotiondetectors.model import * # type: ignore

''' Instantiate the model '''
objModel = instancing_model('Backbonev2')

''' Create an image stream reader '''
objIptStream = ImgstreamReader(os.path.join(gitCodePath, 'demodata', 'imgstream', 'DemoFig*.jpg'))

''' Alternatively, uncomment the following options for different inputs: '''
# objIptStream = ImgstreamReader((os.path.join(gitCodePath, 'demodata', 'imgstream', 'DemoFig*.jpg'), 10,))
# objIptStream = ImgstreamReader((os.path.join(itCodePath, 'demodata', 'imgstream', 'DemoFig*.jpg'), 10, 100 ))

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
    # Read the next frame from the image stream
    grayImg, colorImg = objIptStream.get_next_frame()
    
    # Perform inference using the objModel
    result = inference(objModel, grayImg)
    
    # Display the result
    objVisualize.show_result(colorImg, result)
