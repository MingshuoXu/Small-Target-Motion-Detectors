import os
import sys
import cv2

# Get the full path of this file
filePath = os.path.abspath(__file__)
# Find the index of '/smalltargetmotiondetectors/' in the file path
indexPath = filePath.find(os.path.join(os.sep, 'smalltargetmotiondetectors'))
# Add the path to the package containing the models
sys.path.append(filePath[:indexPath+len('/smalltargetmotiondetectors/')])

from util.iostream import VidstreamReader

# Create test object
test_obj = VidstreamReader()
print('import success')
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
while test_obj.hasFrame:
    garyImg, colorImg = test_obj.get_next_frame()
    cv2.imshow('Image', colorImg)
    cv2.waitKey(100)

