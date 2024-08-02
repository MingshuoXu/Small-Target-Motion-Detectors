# demo_vidstream
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import json

# Get the full path of this file
filePath = os.path.realpath(__file__)
# Find the index of '/+smalltargetmotiondetectors/'
# in the file path
indexPath = filePath.rfind(os.path.sep + 'smalltargetmotiondetectors' + os.path.sep)
# Add the path to the package containing the models
sys.path.append(filePath[:indexPath])

from smalltargetmotiondetectors.api import *


def inference_and_evaluate_task(modelName, 
                                inputpath, 
                                inputType = 'ImgstreamReader', 
                                groundTruth = None,
                                gTError = 1,
                                startFrame = 0, 
                                endFrame = None, 
                                savePath1 = None,
                                savePath2 = None,
                                **kwargs):
    
    '''inference'''
    modelOpt, modelDire = inference_task(modelName, inputpath, inputType, startFrame, endFrame, **kwargs)
    # save
    save_as_json(savePath1, modelOpt, modelDire)

    '''evaluate'''
    rocFig, AUC, mR, RPIList, FPPIList, thresholdList = evaluate_task(modelOpt, groundTruth, gTError, startFrame, endFrame)
    # save
    rocFig.savefig('roc_curve.png')  # Save as PNG file
    save_as_json(savePath2, AUC, mR, RPIList, FPPIList, thresholdList)

    return rocFig, AUC, mR

def save_as_json(file_name='output.json', *args, ):
    """
    Save multiple arguments as a JSON file.

    Parameters:
    - file_name (str): The name of the JSON file to save the data. Defaults to 'output.json'.
    - *args: The data to be saved. Can be multiple objects of any type.
    """
    # Create a dictionary to hold all data
    data = {}
    
    # Generate unique keys for each argument
    for i, arg in enumerate(args):
        key = f"data_{i+1}"
        data[key] = arg
    
    # Ensure the file extension is '.json'
    if not file_name.endswith('.json'):
        file_name += '.json'
    
    # Save data to JSON file
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    with open(os.path.join('C:\\Users\\mings\\Desktop', 'temp_result', 'gt.json'), 'r') as file:
        data = json.load(file)


    modelName = 'ESTMD'
    inputpath = os.path.join('D:\\STMD_Dataset\\PanoramaStimuli\\BV-250-Leftward',
        'SingleTarget-TW-5-TH-5-TV-300-TL-0-Rightward-Amp-15-Theta-0-TemFre-2-SamFre-1000',
        'PanoramaStimuli*.tif')
    inputType = 'ImgstreamReader'
    groundTruth = data['groundTruth']
    gTError = 1
    startFrame = 1
    endFrame = 500
    savePath1 = os.path.join('C:\\Users\\mings\\Desktop', 'temp_result', 'opt1.json')
    savePath2 = os.path.join('C:\\Users\\mings\\Desktop', 'temp_result', 'opt2.json')

    rocFig, AUC, mR = inference_and_evaluate_task(modelName, 
                                                    inputpath, 
                                                    inputType, 
                                                    groundTruth,
                                                    gTError,
                                                    startFrame, 
                                                    endFrame, 
                                                    savePath1,
                                                    savePath2,
                                                    sigma1 = 1)
    
    plt.show()