# demo_vidstream
import matplotlib.pyplot as plt
import numpy as np
import time

from . import (instancing_model, inference)
from ..model import *
from ..util.iostream import (ImgstreamReader, VidstreamReader)
from ..util.evaluate_module import (get_ROC_curve_data, compute_AUC, 
                                    get_thres_recall_data, compute_AR,
                                    get_P_R_curve_data, compute_AP, )
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
    objModel.set_para(**kwargs)
    # init
    objModel.init_config()

    totalRunningTime = 0
    results = []
    directions = []
    ''' Run '''
    while objIptStream.hasFrame:
        # Read the next frame from the video stream
        grayImg, _ = objIptStream.get_next_frame()
        
        # Perform inference using the model
        tic0 = time.time()
        result = inference(objModel, grayImg)
        totalRunningTime += time.time() - tic0

        # postprocessing

        # response
        response = result['response']
        if np.max(response) == 0:
            results.append([])
            continue
        response = objNMS.nms(result['response'])
        maxOpt = np.max(response)
        if maxOpt > 0:
            response /= np.max(response)
            responseListType = matrix_to_sparse_list(response)
        else:
            responseListType = []
        results.append(responseListType)          

        # direction
        direction  = result['direction']
        if (direction is not None) and len(direction) and len(responseListType):
            directionListType = [[y, x, direction[x, y]] for y, x, _ in responseListType]
        else:
            directionListType = []
        directions.append(directionListType)

    return results, directions, totalRunningTime


def evaluate_task(modelOpt, groundTruth, aucPara = 40, gTError = 1, startFrame = 0, endFrame = None, plotFigures=True):
    

    ''' ROC curve Part'''
    # get ROC data
    RPIList, FPPIList, _ = get_ROC_curve_data(modelOpt, 
                                              groundTruth, 
                                              rangeOfFPPI = [0, aucPara], 
                                              gTError = gTError,
                                              startFrame = startFrame,
                                              endFrame = endFrame)
    
    # calculate AUC
    rocOfAUC = compute_AUC(RPIList, FPPIList, rangeOfFPPI=[0, aucPara])

    # plot ROC curve
    if plotFigures:
        rocFig, ax1 = plt.subplots()
        ax1.plot(FPPIList, RPIList)
        ax1.set_xlim(0, aucPara)
        ax1.set_ylim(0, 1)

        ax1.set_xlabel('False Positive Rate (FPPI)')
        ax1.set_ylabel('Recall (RPI)')
        ax1.set_title('ROC Curve')

    ''' mR Part'''
    # get meanRecall data
    RPIList1, thresholdList1 = get_thres_recall_data(modelOpt, 
                                                     groundTruth,
                                                     gTError = gTError,
                                                     startFrame = startFrame, 
                                                     endFrame = endFrame)
    
    # calculate mean Recall
    AR = compute_AR(RPIList1, thresholdList1, rangeOfThreshold=[0.5, 1])
    
    # plot mR curve
    if plotFigures:
        mRFig, ax2 = plt.subplots()
        ax2.plot(thresholdList1, RPIList1)
        ax2.set_xlim(0.5, 1)
        ax2.set_ylim(0, 1)

        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Recall')
        ax2.set_title('Threshold-Recall Curve')

    ''' P-R curve Part'''
    # get meanRecall data
    rList2, pList2, _ = get_P_R_curve_data(modelOpt, 
                                        groundTruth,
                                        intervalOfRecall = 0.02,
                                        gTError = gTError,
                                        startFrame = startFrame, 
                                        endFrame = endFrame)
    
    # calculate mean Recall
    AP = compute_AP(rList2, pList2)

    # plot mR curve
    if plotFigures:
        PRFig, ax3 = plt.subplots()
        ax3.plot(rList2, pList2)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)

        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('P-R Curve')
    
    if plotFigures:
        figHandle = {'ROC': rocFig,
                    'mR':  mRFig,
                    'PR':  PRFig,
                    }

        return rocOfAUC, AR, AP, figHandle
    else:
        return rocOfAUC, AR, AP

