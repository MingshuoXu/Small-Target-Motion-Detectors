import os
import sys
import time

# Get the full path of this file
filePath = os.path.abspath(__file__)
# Find the index of '/smalltargetmotiondetectors/' in the file path
indexPath = filePath.find(os.path.join(os.sep, 'smalltargetmotiondetectors'))
# Add the path to the package containing the models
sys.path.append(filePath[:indexPath+len('/smalltargetmotiondetectors/')])


import numpy as np
from numpy.random import default_rng
import unittest

from util.matrixnms import MatrixNMS, sort_nms, conv2_nms, bubble_nms, greedy_nms

class TestMatrixNMS(unittest.TestCase):
    def setUp(self):
        self.maxRS = 5
        self.matrix_nms = MatrixNMS(maxRegionSize=self.maxRS, method='auto') 

        rng = default_rng(42)
        self.inputMatrix = rng.random((250, 500))
        self.expected_output = self.matrix_nms.nms(self.inputMatrix)

    def test_sort_nms(self):
        output_matrix = sort_nms(self.inputMatrix, self.maxRS)
        self.assertTrue(np.array_equal(output_matrix, self.expected_output))

    def test_conv2_nms(self):
        output_matrix = conv2_nms(self.inputMatrix, self.maxRS)
        self.assertTrue(np.array_equal(output_matrix, self.expected_output))

    def test_bubble_nms(self):
        output_matrix = bubble_nms(self.inputMatrix, self.maxRS)
        self.assertTrue(np.array_equal(output_matrix, self.expected_output))

    def test_greedy_nms(self):
        output_matrix = greedy_nms(self.inputMatrix, self.maxRS)
        self.assertTrue(np.array_equal(output_matrix, self.expected_output))

def test_in_diff_size():
    intM = 250
    intN = 500
    listMaxRegionSize = range(1, 21)
    recordTime1 = np.zeros(len(listMaxRegionSize))
    recordTime2 = np.zeros(len(listMaxRegionSize))
    recordTime3 = np.zeros(len(listMaxRegionSize))
    recordTime4 = np.zeros(len(listMaxRegionSize))
    recordTime5 = np.zeros(len(listMaxRegionSize))

    for iList, maxRegionSize in enumerate(listMaxRegionSize):
        print(f'running... , maxRegionSize: {maxRegionSize}')
        nTimes = 20
        randInput = np.random.rand(intM, intN)

        # bubble method
        obj1 = MatrixNMS(maxRegionSize, 'bubble')
        timeTic0 = time.time()
        for k in range(nTimes):
            opt1 = obj1.nms(randInput)
        recordTime1[iList] = time.time() - timeTic0

        # conv2 method
        obj2 = MatrixNMS(maxRegionSize, 'conv2')
        timeTic0 = time.time()
        for k in range(nTimes):
            opt2 = obj2.nms(randInput)
        recordTime2[iList] = time.time() - timeTic0

        # greedy method
        obj3 = MatrixNMS(maxRegionSize, 'greedy')
        timeTic0 = time.time()
        for k in range(nTimes):
            opt3 = obj3.nms(randInput)
        recordTime3[iList] = time.time() - timeTic0

        # sort method
        obj4 = MatrixNMS(maxRegionSize, 'sort')
        timeTic0 = time.time()
        for k in range(nTimes):
            opt4 = obj4.nms(randInput)
        recordTime4[iList] = time.time() - timeTic0

        # auto method
        obj5 = MatrixNMS(maxRegionSize, 'auto')
        _ = obj5.nms(randInput)  # pre-train for auto
        timeTic0 = time.time()
        for k in range(nTimes):
            opt5 = obj5.nms(randInput)
        recordTime5[iList] = time.time() - timeTic0

    print(f'\n Matrix size : ({intM},{intN})\n')
    print('maxRegionSize\tbubble\tconv2\tgreedy\tsort\tauto')
    for ii in range(len(listMaxRegionSize)):
        print(f'\t{listMaxRegionSize[ii]}\t{recordTime1[ii]:.2f}\t{recordTime2[ii]:.2f}\t{recordTime3[ii]:.2f}\t{recordTime4[ii]:.2f}\t{recordTime5[ii]:.2f}')

if __name__ == '__main__':

    # test_in_diff_size()

    unittest.main()
