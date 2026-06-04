import os
import sys
import cv2

# Get the full path of this file
filePath = os.path.abspath(__file__)
# Find the index of '/smalltargetmotiondetectors/' in the file path
indexPath = filePath.find(os.path.join(os.sep, 'smalltargetmotiondetectors'))
# Add the path to the package containing the models
sys.path.append(filePath[:indexPath+len('/smalltargetmotiondetectors/')])

from util.iostream import Visualization, VidstreamReader

# Create test object
objIpt = VidstreamReader(
    os.path.join(filePath[:indexPath-7], 'demodata', 'RIST_GX010290.mp4')
    )
print('import success')
objOpt = Visualization()
objOpt.create_fig_handle()

# Run inference
while objIpt.hasFrame and objOpt.hasFigHandle:
    # Get the next frame from the input source
    grayImg, colorImg = objIpt.get_next_frame()
    
    # Visualize the result
    objOpt.show_result(colorImg)    

