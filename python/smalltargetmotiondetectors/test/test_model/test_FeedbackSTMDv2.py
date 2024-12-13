import os
import sys

# Add the path to the package containing the models
filePath = os.path.realpath(__file__)
pyPackagePath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(filePath))))
gitCodePath = os.path.dirname(pyPackagePath)
sys.path.append(pyPackagePath)
from smalltargetmotiondetectors.api import get_visualize_handle, inference # type: ignore
from smalltargetmotiondetectors.util.iostream import VidstreamReader, ImgstreamReader # type: ignore
from smalltargetmotiondetectors.model.version2_model import FeedbackSTMDv2 # type: ignore


''' Model instantiation '''
objModel = FeedbackSTMDv2()

''' Input '''
# Demo video (RIST)
hSteam = VidstreamReader(os.path.join(gitCodePath, 'demodata', 'simulatedVideo0_compressed2_250Hz.mp4'))
# hSteam = VidstreamReader(os.path.join(gitCodePath, 'demodata', 'RIST_GX010290_orignal_240Hz.mp4'))
# hSteam = VidstreamReader(os.path.join('D:/', 'STMD_Dataset', 'Real-World-Scence-Material', 'From_Dr_YiZheng', 'fly_bird_and_fish.mp4'))

''' Get visualization handle '''
hVisual = get_visualize_handle(objModel.__class__.__name__)

''' Initialize the model '''
# set the parameter list
objModel.set_para()
# print the parameter list
objModel.print_para()
# init
objModel.init_config()

# Run inference
while hSteam.hasFrame and hVisual.hasFigHandle:
    # Get the next frame from the input source
    grayImg, colorImg = hSteam.get_next_frame()
    
    # Perform inference using the model
    result = inference(objModel, grayImg)
    
    # Visualize the result
    hVisual.show_result(colorImg, result)
