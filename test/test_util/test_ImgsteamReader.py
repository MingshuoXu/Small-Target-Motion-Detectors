import os
import sys
import cv2

filePath = os.path.realpath(__file__)
pyPackagePath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(filePath))))
sys.path.append(pyPackagePath)

from smalltargetmotiondetectors.util.iostream import ImgstreamReader

# Create test object
gitCodePath = os.path.dirname(pyPackagePath)
test_obj = ImgstreamReader(os.path.join(gitCodePath, 'demodata', 'imgstream', 'DemoFig*.jpg'))
print('import success')
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
while test_obj.hasFrame:
    garyImg, colorImg = test_obj.get_next_frame()
    cv2.imshow('Image', colorImg)
    cv2.waitKey(100)

