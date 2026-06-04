import os
import sys
import time

# Get the full path of this file
package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                            'src',)
sys.path.append(package_path)


import numpy as np
import cv2
import torch.nn.functional as F
from numpy.random import default_rng
import unittest

from xttmp.util.matrixnms import MatrixNMS, sort_nms, conv2_nms, bubble_nms, greedy_nms

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


def new_nms(matrix, window_size=8, device='cpu'):
    """
    Perform non-maximum suppression on the matrix using maximum_filter for efficiency.

    Args:
        matrix: Input matrix (numpy.ndarray)
        window_size: Neighborhood window size, must be odd
        device: Computing device ('cpu' or 'cuda')

    Returns:
        nms_matrix: Matrix after non-maximum suppression
    """

    if device == 'cpu':
        # Define neighborhood structure (a window of all ones)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size*2+1, window_size*2+1))
        local_max_cv2 = cv2.dilate(matrix, kernel)
        # Calculate local maximum at each position
        nms_matrix = matrix * (matrix == local_max_cv2)
    else:
        # Use max pooling to get local maximum
        local_max = F.max_pool2d(
            matrix, 
            kernel_size=window_size*2+1, 
            stride=1, 
            padding=window_size
        )
        
        nms_matrix = matrix * (matrix == local_max)

    return nms_matrix

def benchmark_nms(func, *args, iterations=10, warmup=2, **kwargs):
    """
    专门的计时函数
    :param iterations: 正式计时的次数
    :param warmup: 预热次数（不计入成绩）
    """
    # 1. 预热，排除系统初始化干扰
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # 2. 正式计时
    start_time = time.perf_counter()
    for _ in range(iterations):
        func(*args, **kwargs)
    end_time = time.perf_counter()
    
    return (end_time - start_time) / iterations


def run_comprehensive_test():
    rng = default_rng(42)
    
    # 测试用例：(矩阵形状, 窗口大小)
    # 涵盖了小尺寸、长方形、标准高清、以及大窗口场景
    test_cases = [
        ((128, 128), 3),
        ((250, 500), 5),
        ((250, 500), 15),
        ((250, 500), 25),
        ((720, 1280), 7),
        ((1024, 1024), 15),
        ((1024, 1024), 25),
        ((2048, 2048), 25),
    ]

    print(f"{'Shape':<15} | {'Win':<5} | {'New NMS (s)':<12} | {' Name '} | {'Legacy (s)':<12} | {'Speedup':<8}")
    print("-" * 65)

    for shape, win_size in test_cases:
        input_matrix = rng.random(shape).astype(np.float32)

        # 1. 验证正确性 (只运行一次)
        expected_output = new_nms(input_matrix, window_size=win_size, device='cpu')
        sort_nms_output = sort_nms(input_matrix, maxRegionSize=win_size)
        conv2_nms_output = conv2_nms(input_matrix, maxRegionSize=win_size)
        bubble_nms_output = bubble_nms(input_matrix, maxRegionSize=win_size)
        greedy_nms_output = greedy_nms(input_matrix, maxRegionSize=win_size)
        
        # 使用 almost_equal 防止浮点数微小精度差异导致的 assert 失败
        try:
            np.testing.assert_array_almost_equal(sort_nms_output, expected_output, decimal=5)
            np.testing.assert_array_almost_equal(conv2_nms_output, expected_output, decimal=5)
            np.testing.assert_array_almost_equal(bubble_nms_output, expected_output, decimal=5)
            np.testing.assert_array_almost_equal(greedy_nms_output, expected_output, decimal=5)
            status = "PASS"
        except AssertionError:
            status = "FAIL"

        if status == "FAIL":
            print(f"❌ Validation failed for shape {shape}")
            continue

        # 2. 性能测试 (多次运行取平均值)
        # 测试新方法
        avg_new = benchmark_nms(new_nms, input_matrix, window_size=win_size, device='cpu')
        
        # 测试旧方法 (注意：MatrixNMS 需要封装一下，避免重复创建对象影响计时)
        avg_sort = benchmark_nms(sort_nms, input_matrix, maxRegionSize=win_size)
        avg_conv2 = benchmark_nms(conv2_nms, input_matrix, maxRegionSize=win_size)
        avg_bubble = benchmark_nms(bubble_nms, input_matrix, maxRegionSize=win_size)
        avg_greedy = benchmark_nms(greedy_nms, input_matrix, maxRegionSize=win_size)

        # 3. 计算加速比
        speedup_sort = avg_sort / avg_new
        speedup_conv2 = avg_conv2 / avg_new
        speedup_bubble = avg_bubble / avg_new
        speedup_greedy = avg_greedy / avg_new

        print(f"{str(shape):<15} | {win_size:<5} | {avg_new:<12.5f} | {" sort "} | {avg_sort:<12.5f} | {speedup_sort:.2f}x")
        print(f"{'':<15} | {'':<5} | {'':<12} | {"conv2 "} |  {avg_conv2:<12.5f} | {speedup_conv2:.2f}x")
        print(f"{'':<15} | {'':<5} | {'':<12} | {"bubble"} |  {avg_bubble:<12.5f} | {speedup_bubble:.2f}x")
        print(f"{'':<15} | {'':<5} | {'':<12} | {"greey "} |  {avg_greedy:<12.5f} | {speedup_greedy:.2f}x")



if __name__ == '__main__':

    # test_in_diff_size()

    # unittest.main()
    
    # 在测试循环中调用
    run_comprehensive_test()

