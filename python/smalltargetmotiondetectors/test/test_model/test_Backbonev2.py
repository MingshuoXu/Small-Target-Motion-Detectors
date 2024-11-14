import os
import sys
import time

# Get the full path of this file
filePath = os.path.realpath(__file__)
# Find the index of '/+smalltargetmotiondetectors/'
indexPath = filePath.rfind('smalltargetmotiondetectors')
# Add the path to the package containing the models
import_path = filePath[:indexPath]
sys.path.append(import_path)

from smalltargetmotiondetectors.api import (instancing_model, get_visualize_handle, inference) # type: ignore
from smalltargetmotiondetectors.util.iostream import VidstreamReader, ImgstreamReader # type: ignore
from smalltargetmotiondetectors.util.compute_module import matrix_to_sparse_list # type: ignore

''' Model instantiation '''
objModel = instancing_model('Backbonev2')

''' Input '''
# Demo video (RIST)
hSteam = VidstreamReader(os.path.join(filePath[:indexPath-7], 'demodata', 'RIST_GX010290_orignal_240Hz.mp4'))
# hSteam = VidstreamReader(os.path.join(filePath[:indexPath-7], 'demodata', 'RIST_GX010290_compressed2_60Hz.mp4'))
# hSteam = VidstreamReader(os.path.join(filePath[:indexPath-7], 'demodata', 'simulatedVideo0_compressed2_250Hz.mp4'))


''' Get visualization handle '''
hVisual = get_visualize_handle(objModel.__class__.__name__)

''' Initialize the model '''
# set the parameter list
objModel.set_para()
# print the parameter list
objModel.print_para()
# init
objModel.init_config()


'''Run inference'''
while hSteam.hasFrame and hVisual.hasFigHandle:

    # Get the next frame from the input source
    grayImg, colorImg = hSteam.get_next_frame()
    
    # Perform inference using the model
    result = inference(objModel, grayImg)
    # result['response'] = matrix_to_sparse_list(result['response'])

    # # direction
    # direction  = result['direction']
    # if (direction is not None) and len(direction) and len(result['response']):
    #     directionListType = [[y, x, direction[x, y]] for y, x, _ in result['response']]
    # else:
    #     directionListType = []
    # result['direction'] = directionListType
    
    # Visualize the result
    hVisual.show_result(colorImg, result)



