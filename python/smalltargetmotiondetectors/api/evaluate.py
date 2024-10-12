# demo_vidstream
import matplotlib.pyplot as plt
import numpy as np

from . import (instancing_model, inference)
from ..model import *
from ..util.iostream import (ImgstreamReader, VidstreamReader)
from ..util.evaluate_module import (get_ROC_curve_data, compute_AUC, compute_meanRecall)
from ..util.matrixnms import MatrixNMS
from ..util.compute_module import matrix_to_sparse_list


def inference_task(modelName, 
                   inputpath, 
                   inputType = 'ImgstreamReader', 
                   startFrame = 0, 
                   endFrame = None, 
                   **kwargs):
    ''' Instantiate the model '''
    objModel = instancing_model(modelName)

    ''' Dynamically create a video stream reader or other input type '''
    inputModule = globals().get(inputType)
    if inputModule is None:
        raise ValueError(f"Unknown inputType: {inputType}")

    objIptStream = inputModule(inputpath, startFrame, endFrame)

    objNMS = MatrixNMS(15)

    ''' Initialize the model '''
    # set the parameter list
    objModel.set_parameter(**kwargs)
    # init
    objModel.init_config()

    results = []
    directions = []
    ''' Run '''
    while objIptStream.hasFrame:
        # Read the next frame from the video stream
        grayImg, _ = objIptStream.get_next_frame()
        
        # Perform inference using the model
        result = inference(objModel, grayImg)

        # postprocessing
        response = result['response']
        if np.max(response) == 0:
            results.append([])
            continue
        response = objNMS.nms(result['response'])
        response /= np.max(response)
        responseListType = matrix_to_sparse_list(response)
        results.append(responseListType)          

        direction  = result['direction']
        if direction is not None and len(direction):
            directionListType = [[r, c, direction[r,c]] for r, c, _ in responseListType]
            directions.append(directionListType)

    return results,directions


def evaluate_task(modelOpt, groundTruth, gTError = 1, startFrame = 0, endFrame = None):

    RPIList, FPPIList, thresholdList = get_ROC_curve_data(modelOpt, 
                                                          groundTruth, 
                                                          rangeOfFPPI = [0, 1], 
                                                          gTError = gTError,
                                                          startFrame = startFrame,
                                                          endFrame = endFrame)
    # print ROC curve
    fig, ax = plt.subplots()
    ax.plot(FPPIList, RPIList)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xlabel('Recall (RPI)')
    ax.set_ylabel('False Positive Rate (FPPI)')
    ax.set_title('ROC Curve')

    # calculate AUC
    AUC = compute_AUC(RPIList, FPPIList, rangeOfFPPI=[0, 1])
    # calculate mean Recall
    mR = compute_meanRecall(RPIList, thresholdList, rangeOfThreshold=[0.5, 1])

    return fig, AUC, mR, RPIList, FPPIList, thresholdList

