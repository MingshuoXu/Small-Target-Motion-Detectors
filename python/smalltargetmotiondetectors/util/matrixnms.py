import time
import copy

import numpy as np

class MatrixNMS:
    """
    Non-Maximum Suppression in Matrix

    Properties:
        - maxRegionSize: The size of the region for maximum operation.
        - method: The method used for non-maximum suppression.

    Methods:
        - nms: Performs non-maximum suppression on the input matrix.
        - select_auto_method: Automatically selects the method based on input matrix size.
        - sort_nms: Performs non-maximum suppression using sorting method.
        - conv2_nms: Performs non-maximum suppression using conv2 method.
        - bubble_nms: Performs non-maximum suppression using bubble method.
        - greedy_nms: Performs non-maximum suppression using greedy method.
        - mapping_auto_method: Maps auto method based on key-value pairs.
    """

    _mappingAutoMethod = {}

    def __init__(self, maxRegionSize=5, method=None):
        """
        Constructor method
        """
        if method is None:
            if maxRegionSize > 3:
                method = 'sort'
            else:
                method = 'conv2'
        self.maxRegionSize = maxRegionSize
        self.method = method
        self.nullAutoMethod = True
        if not isinstance(maxRegionSize, int) or maxRegionSize <= 0:
            raise ValueError("maxRegionSize must be a positive integer.")
        if method is not None and method not in ['bubble', 'conv2', 'greedy', 'sort', 'auto']:
            raise ValueError("method must be one of 'bubble', 'conv2', 'greedy', 'sort', or 'auto'.")

    def nms(self, inputMatrix):
        """
        Performs non-maximum suppression based on the selected method
        """
        maxRS = self.maxRegionSize

        if self.method == 'auto':
            if self.nullAutoMethod:
                # If auto method is not determined yet
                M, N = inputMatrix.shape
                # Determine auto method based on input matrix size
                self.autoMethod = self.mapping_auto_method(f'{M}-{N}-{maxRS}')
                if not self.autoMethod:
                    # Select auto method if not determined before
                    self.select_auto_method(inputMatrix)
                    # Save auto method in mapping
                    self.mapping_auto_method(f'{M}-{N}-{maxRS}', self.autoMethod)
                self.nullAutoMethod = False

        else:
            self.autoMethod = self.method

        if self.autoMethod == 'conv2':
            outputMartix = conv2_nms(inputMatrix, maxRS)
        
        elif self.autoMethod == 'sort':
            outputMartix = sort_nms(inputMatrix, maxRS)

        elif self.autoMethod == 'greedy':
            outputMartix = greedy_nms(inputMatrix, maxRS)

        elif self.autoMethod == 'bubble':
            outputMartix = bubble_nms(inputMatrix, maxRS)                   

        else:
            # Error for invalid method
            raise ValueError("method must be 'sort', 'bubble', 'conv2', or 'greedy'")

        return outputMartix

    def select_auto_method(self, inputMatrix):
        """
        Selects the most efficient method for non-maximum suppression automatically
        """
        nTimes = 3
        maxRS = self.maxRegionSize
        # Measure the time taken by each method and repeat the process multiple times
        timeBubble, timeConv2, timeGreedy, timeSort = 0, 0, 0, 0
        for methodIdx in range(1, 5):
            for iT in range(nTimes):
                wouldNmsMatrix = inputMatrix

                if iT > 0:
                    timeTic = time.time()
                if methodIdx == 1:
                    bubble_nms(wouldNmsMatrix, maxRS)
                elif methodIdx == 2:
                    conv2_nms(wouldNmsMatrix, maxRS)
                elif methodIdx == 3:
                    greedy_nms(wouldNmsMatrix, maxRS)
                elif methodIdx == 4:
                    sort_nms(wouldNmsMatrix, maxRS)

            timeElapsed = time.time() - timeTic
            if methodIdx == 1:
                timeBubble += timeElapsed
            elif methodIdx == 2:
                timeConv2 += timeElapsed
            elif methodIdx == 3:
                timeGreedy += timeElapsed
            elif methodIdx == 4:
                timeSort += timeElapsed

        # Determine the fastest method
        minTime = min(timeBubble, timeConv2, timeGreedy, timeSort)
        if minTime == timeBubble:
            self.autoMethod = 'bubble'
        elif minTime == timeConv2:
            self.autoMethod = 'conv2'
        elif minTime == timeGreedy:
            self.autoMethod = 'greedy'
        elif minTime == timeSort:
            self.autoMethod = 'sort'
        else:
            raise ValueError('Invalid method index.')

        return self.autoMethod

    @classmethod
    def mapping_auto_method(cls, map_key, add_map_value=None):
        # If no added value is passed, it represents a query operation
        if add_map_value is None:
            return cls._mappingAutoMethod.get(map_key)
        else:
            cls._mappingAutoMethod[map_key] = add_map_value


def conv2_nms(inputMatrix, maxRegionSize):
    """
    Performs non-maximum suppression using conv2 method
    """
    M, N = inputMatrix.shape
    outputMatrix = copy.deepcopy(inputMatrix)  # Initialize output matrix

    # Iterate over the region defined by maxRegionSize
    for rr in range(-maxRegionSize, maxRegionSize + 1):
        for cc in range(-maxRegionSize, maxRegionSize + 1):
            # Define the regions to extract from the input matrix
            rr1 = slice(max(0, 0 + rr), min(M, M + rr))
            cc1 = slice(max(0, 0 + cc), min(N, N + cc))
            rr2 = slice(max(0, 0 - rr), min(M, M - rr))
            cc2 = slice(max(0, 0 - cc), min(N, N - cc))

            # Extract the submatrices from the output matrix and input matrix
            temp = outputMatrix[rr2, cc2]
            inputSubmatrix = inputMatrix[rr1, cc1]

            # Perform element-wise comparison and update the output matrix
            outputMatrix[rr2, cc2] = temp * (temp >= inputSubmatrix)

    return outputMatrix


def sort_nms(inputMatrix, maxRegionSize):
    """
    Performs non-maximum suppression using sorting method
    """
    M, N = inputMatrix.shape
    indexI, indexJ = np.where(inputMatrix > 0)
    valueInputMatrix = inputMatrix[indexI, indexJ]
    indexSub = np.argsort(valueInputMatrix)[::-1]
    indexI = indexI[indexSub]
    indexJ = indexJ[indexSub]

    n = len(indexI)
    isNotSuppress = np.ones((M, N), dtype=bool)

    # Calculate boundaries for local region once
    maxRegion = maxRegionSize
    x_lower = np.maximum(0, indexI - maxRegion)
    x_upper = np.minimum(M, indexI + maxRegion + 1)
    y_lower = np.maximum(0, indexJ - maxRegion)
    y_upper = np.minimum(N, indexJ + maxRegion + 1)

    for idxI in range(n):
        x = indexI[idxI]
        y = indexJ[idxI]
        if isNotSuppress[x, y]:
            internalX = slice(x_lower[idxI], x_upper[idxI])
            internalY = slice(y_lower[idxI], y_upper[idxI])
            isNotSuppress[internalX, internalY] = False

            # Check if the current pixel is the maximum in its local region
            maxLocalValue = np.max(inputMatrix[internalX, internalY])
            if inputMatrix[x, y] == maxLocalValue:
                isNotSuppress[x, y] = True

    # Apply non-maximum suppression
    outputMatrix = inputMatrix * isNotSuppress

    return outputMatrix


def bubble_nms(inputMatrix, maxRegionSize):
    """
    Performs non-maximum suppression using bubble method
    """
    M, N = inputMatrix.shape
    outputMatrix = np.zeros((M, N))  # Initialize output matrix
    copyInputMatrix = copy.deepcopy(inputMatrix)  # Create a copy of input matrix

    # Continue the process until maxValue drops below a threshold
    while np.max(copyInputMatrix) > 1e-16:
        # Find the maximum value and its index in the copy input matrix
        maxValue = np.max(copyInputMatrix)
        maxIndex = np.argmax(copyInputMatrix)

        # Convert the linear index of the maximum value to subscripts
        x, y = maxIndex // N, maxIndex % N

        # Define the region around the maximum value
        startX = max(0, x - maxRegionSize)
        endX = min(M, x + maxRegionSize + 1)
        startY = max(0, y - maxRegionSize)
        endY = min(N, y + maxRegionSize + 1)

        # Find the maximum value within the defined region
        maxLocalValue = np.max(inputMatrix[startX:endX, startY:endY])

        # Check if the maximum value in the region is equal to the maximum value in the input matrix
        if maxValue == maxLocalValue:
            outputMatrix[x, y] = maxLocalValue  # Set the output matrix value to the maximum value

        # Set the values in the defined region of the copy input matrix to zero
        copyInputMatrix[startX:endX, startY:endY] = 0

    return outputMatrix


def greedy_nms(inputMatrix, maxRegionSize):
    """
    Performs non-maximum suppression using the greedy method
    """
    M, N = inputMatrix.shape
    gIptMatrix = inputMatrix
    gIsNotSupp = np.ones((M, N), dtype=bool)

    gM = M
    gN = N
    gMaxRS = maxRegionSize
    gOptMatrix = np.zeros((M, N))

    # Define a nested function to recursively find the maximum and suppress
    def findMax2Supp(x, y):
        nonlocal gIsNotSupp, gIptMatrix, gOptMatrix, gM, gN, gMaxRS

        localInternalX = slice(max(0, x - gMaxRS), min(gM, x + gMaxRS + 1))
        localInternalY = slice(max(0, y - gMaxRS), min(gN, y + gMaxRS + 1))

        maxIndex = np.argmax(gIptMatrix[localInternalX, localInternalY])

        maxIdX0, maxIdY0 = np.unravel_index(
            maxIndex, 
            (localInternalX.stop - localInternalX.start, localInternalY.stop - localInternalY.start)
        )

        maxIdX = maxIdX0 + localInternalX.start
        maxIdY = maxIdY0 + localInternalY.start

        internalMaxIdX = slice(max(0, maxIdX - gMaxRS), min(gM, maxIdX + gMaxRS + 1))
        internalMaxIdY = slice(max(0, maxIdY - gMaxRS), min(gN, maxIdY + gMaxRS + 1))

        if maxIdX == x and maxIdY == y:
            gOptMatrix[maxIdX, maxIdY] = gIptMatrix[maxIdX, maxIdY]
            gIsNotSupp[internalMaxIdX, internalMaxIdY] = False
        else:
            if gIsNotSupp[maxIdX, maxIdY]:
                findMax2Supp(maxIdX, maxIdY)
            if maxIdX < x or (maxIdX == x and maxIdY < y):
                intersectX = slice(
                    max(localInternalX.start, internalMaxIdX.start),
                    min(localInternalX.stop, internalMaxIdX.stop)
                    )
                intersectY = slice(
                    max(localInternalY.start, internalMaxIdY.start),
                    min(localInternalY.stop, internalMaxIdY.stop)
                    )
                gIsNotSupp[intersectX, intersectY] = False

    # Iterate over each element in the input matrix
    for x in range(M):
        for y in range(N):
            if gIsNotSupp[x, y]:
                findMax2Supp(x, y)

    return gOptMatrix


if __name__ == "__main__":
    from numpy.random import default_rng

    obj = MatrixNMS(maxRegionSize=5, method='auto')
    rng = default_rng(42)
    Ipt =  rng.random((250, 500))
    Opt = obj.nms(Ipt)



