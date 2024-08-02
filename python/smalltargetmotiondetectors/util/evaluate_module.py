

def evaluation_model_by_video(modelOpt: list, 
                              groundTruth: list, 
                              confidenceThreshold: float = 0.5, 
                              gTError: int = 1,
                              ROIThreshold: float = 0.5):
    """
    Evaluates the performance of the detection results over a video sequence.

    Parameters:
    - modelOpt (list of lists): Detection results for each frame in the video. 
      Each item should be in the format:
        - Center indices: [ [[i1, j1, confidence1]], ... ] with shape(totalFrame, numOfTarget, 3)
        - Bounding boxes: [ [[i1, j1, h1, w1, confidence1]], ... ] with shape(totalFrame, numOfTarget, 5)
    - groundTruth (list of lists): Ground truth data for each frame. 
      Each item should be in the format:
        - Center indices: [ [[i1, j1]], ... ] with shape(totalFrame, numOfTarget, 2)
        - Bounding boxes: [ [[i1, j1, h1, w1]], ... ] with shape(totalFrame, numOfTarget, 4)
    - confidenceThreshold (float): Minimum confidence score to consider a prediction.
    - gTError (int or float): Maximum allowable error distance for matching ground truth data.
    - ROIThreshold (float): Threshold for the region of interest.

    Returns:
    - listTP: List of true positives for each frame.
    - listFN: List of false negatives for each frame.
    - listFP: List of false positives for each frame.
    """
    
    # Initialize the metrics lists with zeros
    totalLen = min(len(modelOpt), len(groundTruth))
    listTP = [0 for _ in range(totalLen)]
    listFN = [0 for _ in range(totalLen)]
    listFP = [0 for _ in range(totalLen)]

    # Iterate over each frame
    for idx in range(totalLen):
        # Check if current frame data is available
        if modelOpt[idx] and groundTruth[idx]:
            # Compute metrics for the current frame
            TP, FN, FP = compute_metrics_by_frame(modelOpt[idx], 
                                                  groundTruth[idx], 
                                                  confidenceThreshold, 
                                                  gTError,
                                                  ROIThreshold)
        else:
            # If data for the frame is not available, set metrics to zero
            TP, FN, FP = 0, 0, 0

        # Store the metrics for the current frame
        listTP[idx] = TP
        listFN[idx] = FN
        listFP[idx] = FP

    return listTP, listFN, listFP


def compute_metrics_by_frame(prediction, groundTruth, confidenceThreshold=0.5, gTError=1, ROIThreshold=0.5):
    """
    Computes evaluation metrics based on predicted and ground truth data.

    Parameters:
    - prediction (list of lists or tuples): Predicted data from an image/frame. 
      Each item should have a confidence score as the last element.
    - groundTruth (list of lists or tuples): Ground truth data. 
      Format depends on the type of data.
    - confidenceThreshold (float): Minimum confidence score to consider a prediction.
    - gTError (int or float): Error margin for matching ground truth data.
    - ROIThreshold (float): Threshold for region of interest.

    Returns:
    - TP (int): Number of true positives.
    - FN (int): Number of false negatives.
    - FP (int): Number of false positives.
    """
    
    TP = FP = 0
    
    # Determine the matching function based on the dimensions of the prediction and ground truth data
    if len(prediction[0]) == 3:
        if len(groundTruth[0]) == 2:
            matchFun = match_two_dots
        elif len(groundTruth[0]) == 4:
            matchFun = match_dot_in_bbox
    elif len(prediction[0]) == 5:
        if len(groundTruth[0]) == 2:
            matchFun = match_bbox_cover_dot
        elif len(groundTruth[0]) == 4:
            matchFun = match_two_bboxs
    
    # Initialize a list to track which ground truth items are false negatives
    isGTaFN = [True for _ in range(len(groundTruth))]
    
    # Iterate through predictions
    for pre in prediction:
        if pre[-1] < confidenceThreshold:
            continue  # Skip predictions with confidence below the threshold

        isTP = False
        for i, gT in enumerate(groundTruth):
            if matchFun(pre, gT, gTError, ROIThreshold):
                isTP = True
                isGTaFN[i] = False
                break  # Exit the loop once a match is found
        
        if isTP: 
            TP += 1
        else:
            FP += 1
    
    # Calculate false negatives
    FN = isGTaFN.count(True)

    return TP, FN, FP


def match_two_dots(dot1, dot2, gTError, _):
    """
    Determines if two points are within a specified error range of each other.

    Parameters:
    - dot1 (tuple or list): The first point, containing coordinates (x1, y1).
    - dot2 (tuple or list): The second point, containing coordinates (x2, y2).
    - gTError (int or float): The maximum allowable error distance between the points.
    - _ : Placeholder for an unused parameter.

    Returns:
    - bool: True if the points are within the specified error range in all dimensions, otherwise False.
    
    Example:
    >>> match_two_dots((10, 15), (12, 18), 3, None)
    True
    >>> match_two_dots((10, 15), (14, 18), 1, None)
    False
    """

    return all(abs(d1 - d2) <= gTError for d1, d2 in zip(dot1, dot2))


def match_dot_in_bbox(dot1, bbox2, gTError, _):
    """
    Determines if a point is within a bounding box with a specified error range.

    Parameters:
    - dot1 (tuple or list): The point to check, containing coordinates (x, y).
    - bbox2 (tuple or list): The bounding box, specified as (x, y, width, height).
    - gTError (int or float): The maximum allowable error distance.
    - _ : Placeholder for an unused parameter.

    Returns:
    - bool: True if the point is within the bounding box extended by the error range, otherwise False.
    
    Example:
    >>> match_dot_in_bbox((5, 5), (4, 4, 2, 2), 1, None)
    True
    >>> match_dot_in_bbox((5, 5), (4, 4, 1, 1), 0.5, None)
    False
    """
    
    # Check if the point's x coordinate is within the bounding box's x range, including the error margin
    x_within_bbox = dot1[0] >= bbox2[0] - gTError and dot1[0] <= bbox2[0] + bbox2[2] + gTError
    
    # Check if the point's y coordinate is within the bounding box's y range, including the error margin
    y_within_bbox = dot1[1] >= bbox2[1] - gTError and dot1[1] <= bbox2[1] + bbox2[3] + gTError
    
    # Return True if both x and y coordinates are within the extended bounding box, otherwise False
    return x_within_bbox and y_within_bbox


def match_bbox_cover_dot(bbox1, dot2, gTError, _):
    """
    Determines if a point is within a bounding box with a specified error range,
    where the bounding box and point roles are swapped compared to the `match_dot_in_bbox` function.

    Parameters:
    - bbox1 (tuple or list): The bounding box to check, specified as (x, y, width, height).
    - dot2 (tuple or list): The point to check, containing coordinates (x, y).
    - gTError (int or float): The maximum allowable error distance.
    - _ : Placeholder for an unused parameter.

    Returns:
    - bool: True if the point is within the bounding box extended by the error range, otherwise False.
    
    Example:
    >>> match_bbox_cover_dot((4, 4, 2, 2), (5, 5), 1, None)
    True
    >>> match_bbox_cover_dot((4, 4, 2, 2), (6, 6), 0.5, None)
    False
    """
    
    # Use `match_dot_in_bbox` to determine if `dot2` is within `bbox1` extended by `gTError`
    return match_dot_in_bbox(dot2, bbox1, gTError, None)


def match_two_bboxs(bbox1, bbox2, _, ROIThreshold):
    """
    Checks if the ROI between two rectangles is greater than a specified threshold.

    Parameters:
    - bbox1: The first rectangle in the format [x, y, w, h].
    - bbox2: The second rectangle in the format [x, y, w, h].
    - _: Placeholder for an unused parameter.
    - ROIThreshold: The threshold value to compare the ROI against.

    Returns:
    - bool: True if the ROI is greater than the ROIThreshold, otherwise False.
    """
    # Compute the ROI between the two rectangles
    ROI = compute_ROI(bbox1, bbox2)

    # Return True if the ROI is greater than the threshold, otherwise False
    return ROI > ROIThreshold


def compute_ROI(rect1, rect2):
    """
    Computes the region of interest (ROI) between two rectangles.

    Parameters:
    - rect1: A rectangle region in the format [x, y, w, h], where (x, y) is the top-left corner,
             w is the width, and h is the height.
    - rect2: A rectangle region in the format [x, y, w, h], where (x, y) is the top-left corner,
             w is the width, and h is the height.

    Returns:
    - ROI: The region of interest between the two rectangles, defined as the intersection area
           divided by the union of the areas of the two rectangles.
    """
    # Unpack rectangle coordinates and dimensions
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Determine the coordinates of the intersection rectangle
    left = max(x1, x2)
    top = max(y1, y2)
    right = min(x1 + w1, x2 + w2)
    bottom = min(y1 + h1, y2 + h2)

    # Calculate the area of the intersection rectangle
    if left < right and top < bottom:
        intersection_area = (right - left) * (bottom - top)
    else:
        intersection_area = 0

    # Calculate the area of both rectangles
    area1 = w1 * h1
    area2 = w2 * h2

    # Calculate the ROI (Intersection over Union)
    union_area = area1 + area2 - intersection_area
    ROI = intersection_area / union_area if union_area > 0 else 0

    return ROI


def get_RFI_by_fixFPPI(modelOpt: list, 
                       groundTruth: list,
                       aimFPPI: float = 1,  
                       gTError: int = 1,
                       ROIThreshold: float = 0.5,
                       startFrame=0, 
                       endFrame=None):
    """
    Calculates the RFI (Recall Per Image) based on a fixed FPPI (False Positives Per Image) for a given model output.

    Parameters:
    - modelOpt (list of lists): Model output data for each frame. Each item can be:
        - Center indices: [ [[i1, j1, confidence1]], ... ] with shape(totalFrame, numOfTarget, 3)
        - Bounding boxes: [ [[i1, j1, h1, w1, confidence1]], ... ] with shape(totalFrame, numOfTarget, 5)
    - groundTruth (list of lists): Ground truth data for each frame. Each item can be:
        - Center indices: [ [[i1, j1]], ... ] with shape(totalFrame, numOfTarget, 2)
        - Bounding boxes: [ [[i1, j1, h1, w1]], ... ] with shape(totalFrame, numOfTarget, 4)
    - aimFPPI (float): Target value for False Positive Per Image.
    - gTError (int or float): Maximum allowable error distance for matching ground truth data.
    - ROIThreshold (float): Threshold for the region of interest.
    - startFrame (int): Starting frame of the dataset.
    - endFrame (int): Ending frame of the dataset (optional).

    Returns:
    - RFI (float): Recall Per Image corresponding to the aimFPPI.
    """

    # Initialize threshold boundaries and value
    thresholdTop = 1.0
    thresholdDown = 0.0
    thresholdValue = 0.5
    preThresholdValue = 0.0

    while True:
        # Filter data based on the current threshold
        threInput = [[data for data in frame if data[-1] >= thresholdValue] 
                     for frame in modelOpt]

        # Compute metrics
        listTP, listFN, listFP = evaluation_model_by_video(threInput, 
                                                            groundTruth, 
                                                            confidenceThreshold=thresholdValue, 
                                                            gTError=gTError,
                                                            ROIThreshold=ROIThreshold)

        # Calculate RFI and FPPI
        numberAT = sum(listFN[startFrame:endFrame+1]) + sum(listTP[startFrame:endFrame+1])
        # Calculate Recall Per Image (RFI) and False Positive Per Image (FPPI)
        RFI = sum(listTP[startFrame:endFrame+1]) / numberAT if numberAT > 0 else 0
        FPPI = sum(listFP[startFrame:endFrame + 1]) / (endFrame - startFrame + 1) \
            if len(listFP[startFrame:endFrame+1]) > 0 else 0

        # Check for convergence
        if abs(FPPI - aimFPPI) < 0.01 or abs(thresholdValue - preThresholdValue) < 1e-5:
            break
        else:
            # Update thresholds
            if FPPI < aimFPPI:
                thresholdTop = thresholdValue
            else:
                thresholdDown = thresholdValue
            preThresholdValue = thresholdValue
            thresholdValue = (thresholdTop + thresholdDown) / 2

    return RFI


def get_ROC_curve_data( modelOpt: list, 
                        groundTruth: list,
                        rangeOfFPPI: list = [0, 1],
                        gTError: int = 1,
                        ROIThreshold = 0.5,
                        startFrame=0, 
                        endFrame=None):
    """
    Calculates the RFI (Recall Per Image) based on a list of FPPI (False Positives Per Image) for a given model output.

    Parameters:
    - modelOpt: Model output data.
    - groundTruth: Ground truth data.
    - rangeOfFPPI: Range of False Positive Per Image.
    - gTError: Distance error scope for ground truth.
    - ROIThreshold: ROI threshold.
    - startFrame: Starting frame of the dataset.
    - endFrame: Ending frame of the dataset (optional).

    Returns:
    - RPIList: List of Recall Per Image.
    - FPPIList: List of False Positives Per Image.
    - thresholdList: List of thresholds.
    """

    if endFrame is None:
        endFrame = len(modelOpt) - 1

    lowerFPPI, upperFPPI = rangeOfFPPI
    intevalFPPI = (upperFPPI - lowerFPPI) / 20
    errorFPPI = (upperFPPI - lowerFPPI) / 100

    thresholdList = [i * 0.1 for i in range(11)]
    FPPIList = [None for _ in range(11)]
    RPIList = [None for _ in range(11)]
    listMark = [True for _ in range(11)]

    while True:
        idx = next((index for index, value in enumerate(listMark) if value), None)
        if idx is None:
            break

        thresholdValue = thresholdList[idx]

        # Filter data based on the threshold value
        threInput = [[data for data in frame if data[-1] >= thresholdValue] 
                     for frame in modelOpt]

        # Evaluate the model by video
        listTP, listFN, listFP = evaluation_model_by_video(threInput, 
                                                           groundTruth, 
                                                           confidenceThreshold=thresholdValue, 
                                                           gTError=gTError,
                                                           ROIThreshold=ROIThreshold)

        # Calculate RFI and FPPI
        numberAT = sum(listFN[startFrame:endFrame+1]) + sum(listTP[startFrame:endFrame+1])
        # Calculate Recall Per Image (RFI) and False Positive Per Image (FPPI)
        RFI = sum(listTP[startFrame:endFrame+1]) / numberAT if numberAT > 0 else 0
        FPPI = sum(listFP[startFrame:endFrame + 1]) / (endFrame - startFrame + 1) \
            if len(listFP[startFrame:endFrame+1]) > 0 else 0


        RPIList[idx] = RFI
        FPPIList[idx] = FPPI
        listMark[idx] = False

        if all(not element for element in listMark):
            filteredIdx = [i for i, x in enumerate(FPPIList) if lowerFPPI <= x <= upperFPPI]
            thresholdList0 = [thresholdList[i] for i in filteredIdx]
            FPPIList0 = [FPPIList[i] for i in filteredIdx]
            RPIList0 = [RPIList[i] for i in filteredIdx]
            listMark0 = [listMark[i] for i in filteredIdx]

            # Adjust threshold for lower bound
            if abs(FPPIList[filteredIdx[0]] - lowerFPPI) > errorFPPI:
                if filteredIdx[0] > 0:
                    addLowerThres = (thresholdList[filteredIdx[0]-1] + thresholdList[filteredIdx[0]]) / 2
                else:
                    addLowerThres = thresholdList[filteredIdx[0]] / 2
                thresholdList0 = [addLowerThres] + thresholdList0
                FPPIList0 = [None] + FPPIList0
                RPIList0 = [None] + RPIList0
                listMark0 = [True] + listMark0

            # Adjust threshold for upper bound
            if abs(FPPIList[filteredIdx[-1]] - upperFPPI) > errorFPPI:
                if filteredIdx[-1] < len(thresholdList) - 1:
                    addupperThres = (thresholdList[filteredIdx[-1] + 1] + thresholdList[filteredIdx[-1]]) / 2
                else:
                    addupperThres = (thresholdList[filteredIdx[0]] + 1) / 2
                thresholdList0 = thresholdList0 + [addupperThres]
                FPPIList0 = FPPIList0 + [None]
                RPIList0 = RPIList0 + [None]
                listMark0 = listMark0 + [True]

            # Add intermediate thresholds to reduce FPPI interval
            i = 0
            while i < len(thresholdList0) - 1:
                if abs(FPPIList0[i + 1] - FPPIList0[i]) > intevalFPPI:
                    mean = (thresholdList0[i] + thresholdList0[i + 1]) / 2
                    thresholdList0.insert(i + 1, mean)
                    FPPIList0.insert(i + 1, None)
                    RPIList0.insert(i + 1, None)
                    listMark0.insert(i + 1, True)
                else:
                    i += 1

            thresholdList = thresholdList0
            FPPIList = FPPIList0
            RPIList = RPIList0
            listMark = listMark0

    return RPIList, FPPIList, thresholdList


def get_meanRecall_data(modelOpt: list, 
                        groundTruth: list,
                        rangeOfThreshold: list = [0.5, 1],
                        thresholdInteval: float = 0.01,
                        gTError: int = 1,
                        ROIThreshold = 0.5,
                        startFrame=0, 
                        endFrame=None):
    """
    Calculates the RFI (Recall Per Image) based on list Threshold for a given model output.

    Parameters:
    - modelOpt: List containing the detection results.
    - groundTruth: List containing the ground truth.
    - rangeOfThreshold: List defining the range of thresholds [lowerBound, upperBound].
    - thresholdInteval: Interval between thresholds.
    - gTError: Distance error scope for ground truth.
    - ROIThreshold: ROI threshold.
    - startFrame: Starting frame of the dataset.
    - endFrame: Ending frame of the dataset (optional).

    Returns:
    - RPIList: List of Recall Per Image.
    - thresholdList: List of thresholds.
    """

    if endFrame is None:
        endFrame = len(modelOpt) - 1
    
    lowerBound, upperBound = rangeOfThreshold
    totalLen = int((upperBound - lowerBound) / thresholdInteval)

    thresholdList = [lowerBound + i * thresholdInteval for i in range(totalLen)] + [upperBound]
    RPIList = [None for _ in range(totalLen + 1)]
    
    for idx, thresholdValue in enumerate(thresholdList):
        # Filter data based on the threshold value
        threInput = [[data for data in frame if data[-1] >= thresholdValue] 
                     for frame in modelOpt]

        # Evaluate the model by video
        listTP, listFN, listFP = evaluation_model_by_video(threInput, 
                                                           groundTruth, 
                                                           confidenceThreshold=thresholdValue, 
                                                           gTError=gTError,
                                                           ROIThreshold=ROIThreshold)

        # Calculate RFI and FPPI
        numberAT = sum(listFN[startFrame:endFrame+1]) + sum(listTP[startFrame:endFrame+1])
        # Calculate Recall Per Image (RFI) and False Positive Per Image (FPPI)
        RFI = sum(listTP[startFrame:endFrame+1]) / numberAT if numberAT > 0 else 0

        RPIList[idx] = RFI

    return RPIList, thresholdList


def compute_AUC(RPIList, FPPIList, rangeOfFPPI=[0, 1]):
    """
    Computes the Area Under the Curve (AUC) for the given RPI (Recall Per Image) and FPPI (False Positives Per Image) lists.

    Parameters:
    - RPIList: List of Recall Per Image values.
    - FPPIList: List of False Positives Per Image values.
    - rangeOfFPPI: Range of False Positives Per Image to consider for AUC calculation (default is [0, 1]).

    Returns:
    - AUC: Area Under the Curve.
    """
    lowerFPPI, upperFPPI = rangeOfFPPI

    # Ensure the lists are sorted by FPPIList
    sorted_indices = sorted(range(len(FPPIList)), key=lambda i: FPPIList[i])
    sorted_RPI = [RPIList[i] for i in sorted_indices]
    sorted_FPPI = [FPPIList[i] for i in sorted_indices]

    # Filter the lists based on the specified rangeOfFPPI
    filtered_RPI = []
    filtered_FPPI = []
    for i in range(len(sorted_FPPI)):
        if lowerFPPI <= sorted_FPPI[i] <= upperFPPI:
            filtered_RPI.append(sorted_RPI[i])
            filtered_FPPI.append(sorted_FPPI[i])

    # Compute AUC using the trapezoidal rule
    AUC = 0.0
    for i in range(1, len(filtered_RPI)):
        AUC += (filtered_FPPI[i] - filtered_FPPI[i - 1]) * (filtered_RPI[i] + filtered_RPI[i - 1]) / 2.0

    return AUC


def compute_meanRecall(RPIList, thresholdList, rangeOfThreshold=[0.5, 1]):
    """
    Computes the mean recall based on the given RPI (Recall Per Image) and threshold lists.

    Parameters:
    - RPIList: List of Recall Per Image values.
    - thresholdList: List of threshold values.
    - rangeOfThreshold: Range of threshold values to consider for mean recall calculation (default is [0.5, 1]).

    Returns:
    - meanRecall: Weighted mean recall value.
    """
    lowerThreshold, upperThreshold = rangeOfThreshold

    # Ensure the lists are sorted by thresholdList
    sorted_indices = sorted(range(len(thresholdList)), key=lambda i: thresholdList[i])
    sorted_RPI = [RPIList[i] for i in sorted_indices]
    sorted_thresholds = [thresholdList[i] for i in sorted_indices]

    # Filter the lists based on the specified rangeOfThreshold
    filtered_RPI = []
    filtered_thresholds = []
    for i in range(len(sorted_thresholds)):
        if lowerThreshold <= sorted_thresholds[i] <= upperThreshold:
            filtered_RPI.append(sorted_RPI[i])
            filtered_thresholds.append(sorted_thresholds[i])

    # Compute weighted mean recall using the trapezoidal rule
    weighted_sum = 0.0
    total_weight = 0.0
    for i in range(1, len(filtered_RPI)):
        delta_threshold = filtered_thresholds[i] - filtered_thresholds[i - 1]
        weighted_sum += (filtered_RPI[i] + filtered_RPI[i - 1]) / 2.0 * delta_threshold
        total_weight += delta_threshold

    meanRecall = weighted_sum / total_weight if total_weight != 0 else 0.0

    return meanRecall




