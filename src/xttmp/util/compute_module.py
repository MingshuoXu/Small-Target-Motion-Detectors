import cv2
import numpy as np
import torch
import torch.nn.functional as F


def compute_temporal_conv(iptCell, kernel, pointer=None):
    """
    Computes temporal convolution.

    Parameters:
    - iptCell: A list of arrays where each element has the same dimension.
    - kernel: A vector representing the convolution kernel.
    - headPointer: Head pointer of the input cell array (optional).

    Returns:
    - optMatrix: The result of the temporal convolution.
    """

    # Default value for headPointer
    if pointer is None:
        pointer = len(iptCell) - 1

    # Initialize output matrix
    if iptCell[pointer] is None:
        return None

    # Ensure kernel is a vector
    kernel = np.squeeze(kernel)
    if not np.ndim(kernel) == 1:
        raise ValueError('The kernel must be a vector.')

    # Determine the lengths of input cell array and kernel
    k1 = len(iptCell)
    k2 = len(kernel)
    length = min(k1, k2)

    if isinstance(iptCell[pointer], np.ndarray):
        optMatrix = np.zeros_like(iptCell[pointer])
    elif isinstance(iptCell[pointer], torch.Tensor):
        optMatrix = torch.zeros_like(iptCell[pointer])
    # Perform temporal convolution
    for t in range(length):
        j = (pointer - t) % k1
        if abs(kernel[t]) > 1e-16 and iptCell[j] is not None:
            optMatrix += iptCell[j] * kernel[t]

    return optMatrix


def compute_circularlist_conv(circularCell, temporalKernel):
    """
    Compute the convolution of a circular cell with a temporal kernel.
    
    Args:
    - circularCell: The circular cell data.
    - temporalKernel: The temporal kernel data.
    
    Returns:
    - opt_matrix: The result of the convolution.
    """
    optMatrix = compute_temporal_conv(circularCell, 
                                      temporalKernel, 
                                      circularCell.pointer )
    return optMatrix


def compute_response(ipt, device='cpu'):
    """
    Computes the maximum response from multiple inputs.

    Parameters:
    - ipt: List containing input data.

    Returns:
    - response: Maximum response computed from the inputs.
    """
    if device != 'cpu':
        response = torch.amax(ipt, dim=1, keepdim=True)
    else:
        response = np.max(ipt, axis=0)


    return response


def compute_direction(ipt, device='cpu'):
    """
    Compute the dominant direction given a set of directional responses

    Parameters:
    - ipt: List containing directional responses.

    Returns:
    - direction_opt: Dominant direction computed from the responses.
    """


    if device != 'cpu':
        B, C, H, W = ipt.shape # C = 8 (numDirection)
    
        # 1. 预计算每个通道对应的单位向量角度 (theta)
        # angles = [0, 1/8*2pi, 2/8*2pi, ...]
        angles = torch.linspace(0, 2 * torch.pi, steps=C+1, device=device)[:-1].to(device=device)
        
        # 2. 计算对应的 Cos 和 Sin 权重基准
        # 形状为 [8]，调整为 [1, 8, 1, 1] 以便进行广播乘法
        cos_weight = torch.cos(angles).view(1, C, 1, 1)
        sin_weight = torch.sin(angles).view(1, C, 1, 1)
        
        # 3. 计算加权和 (替代原代码中的 for 循环)
        # ipt * cos_weight 形状仍为 [1, 8, H, W]
        # 对 dim=1 (通道维) 求和，得到 [1, H, W]
        output_cos = torch.sum(ipt * cos_weight, dim=1)
        output_sin = torch.sum(ipt * sin_weight, dim=1)
        
        # 4. 使用 atan2 计算合成方向
        # 结果范围是 (-pi, pi]
        direction_opt = torch.atan2(output_sin, output_cos)
        
        # 5. 调整范围到 [0, 2*pi]
        direction_opt = torch.where(direction_opt < 0, direction_opt + 2 * torch.pi, direction_opt)
        
        # 6. 处理无效像素 (Sin 和 Cos 同时接近 0 的地方)
        # 原代码用 bool 转换，torch 中建议用极小阈值判断，或者直接检查全零
        # 只有当两个分量都非常小时才设为 NaN
        mask = (output_sin == 0) & (output_cos == 0)
        direction_opt[mask] = float('nan')
        direction_opt = direction_opt.unsqueeze(0)  # 去掉批次维度
    else:
        numDirection = len(ipt)
    
        # 1. 预计算每个方向的角度 (theta)
        # 使用 np.linspace 快速生成 [0, 2*pi)
        angles = np.linspace(0, 2 * np.pi, numDirection, endpoint=False)
        
        # 2. 预计算权重向量 (形状为 [numDirection, 1, 1])
        # 增加维度是为了利用广播机制直接与 (C, H, W) 相乘
        cos_weights = np.cos(angles)[:, np.newaxis, np.newaxis]
        sin_weights = np.sin(angles)[:, np.newaxis, np.newaxis]
        
        # 3. 向量化计算加权和 (替代 for 循环)
        # 对第一个轴 (axis=0) 求和，一次性得到所有的 Cos 和 Sin 分量
        outputCos = np.sum(ipt * cos_weights, axis=0)
        outputSin = np.sum(ipt * sin_weights, axis=0)
        
        # 4. 计算方向并调整范围至 [0, 2*pi]
        # np.arctan2 自动处理象限
        direction_opt = np.arctan2(outputSin, outputCos)
        direction_opt %= (2 * np.pi) # 使用取模运算替代 if 判断，更简洁
        
        # 5. 处理无效像素 (Sin 和 Cos 同时为 0 的地方)
        # 注意：浮点数比较建议使用 np.isclose 或设定极小阈值
        # 原代码逻辑：非(Sin且Cos) -> 即(Sin为0 或 Cos为0)
        # 修正逻辑：通常应该是两者都为 0 时才设为 NaN
        invalid_mask = np.isclose(outputSin, 0) & np.isclose(outputCos, 0)
        direction_opt[invalid_mask] = np.nan

    return direction_opt


def slice_matrix_holding_size(iptMatrix, shiftX, shiftY):
    """
    Slice the input matrix while maintaining its size by circular shifting.

    Parameters:
    - input_mat: Input matrix.
    - shiftX: Shift value along the x-axis.
    - shiftY: Shift value along the y-axis.

    Returns:
    - Opt: Sliced matrix holding the original size.
    """
    # Round shift values to integers
    shiftX = round(shiftX)
    shiftY = round(shiftY)

    # Get the size of the input matrix
    m, n = iptMatrix.shape

    # If the shifts exceed the matrix dimensions, return a matrix of zeros
    if abs(shiftX) >= n or abs(shiftY) >= m:
        return np.zeros((m, n))

    # Perform circular shift on the input matrix
    Opt = np.roll(iptMatrix, (shiftX, shiftY), axis=(1, 0))

    # Set the sliced regions to zero based on the shift direction
    if shiftX > 0:
        Opt[:, :shiftX] = 0
    else:
        Opt[:, shiftX:] = 0

    if shiftY > 0:
        Opt[:shiftY, :] = 0
    else:
        Opt[shiftY:, :] = 0

    return Opt


def matrix_to_sparse_list(matrix):
    """
    Convert a matrix to a list of non-zero elements in the format [row, col, value].
    
    Parameters:
    - matrix (numpy.ndarray): The input matrix to be converted.

    Returns:
    - list: A list of non-zero elements in the format [row, col, value].
    """

    import numpy as np

    # Ensure the input is a NumPy array
    matrix = np.array(matrix)
    
    # Get the indices and values of non-zero elements
    rows, cols = np.nonzero(matrix)
    values = matrix[rows, cols]
    
    # Convert numpy types to Python types
    rows = rows.astype(int).tolist()
    cols = cols.astype(int).tolist()
    values = values.astype(float).tolist()
    
    # Combine rows, cols, and values into a list of tuples
    sparseList = [[x, y, v] for y, x, v in zip(rows, cols, values)]
    
    return sparseList


class AreaNMS:
    """
    Area-based Non-Maximum Suppression (NMS).
    Suppresses non-local maximum values using dilation (CPU) or max pooling (GPU).
    """

    def __init__(self, radio=8, device='cpu'):
        """
        Args:
            radio (int): Neighborhood radius. Window size = 2 * radio + 1.
            device (str): Computing device ('cpu' or 'cuda').
        """
        self.device = device
        self.radio = radio
        self.ksize = self.radio * 2 + 1
        
        self._setup()

    def _setup(self):
        self.ksize = self.radio * 2 + 1
        """Initialize device-specific resources."""
        if self.device == 'cpu':
            # Pre-compute structuring element for morphological dilation
            self.cv2_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                                        (self.ksize, self.ksize))

    def process(self, matrix):
        """
        Apply NMS to the input matrix.

        Args:
            matrix (np.ndarray | torch.Tensor): Input heatmap or score map.
                - If CPU: Expected shape (H, W) as numpy array.
                - If CUDA: Expected shape (B, C, H, W) as torch tensor.

        Returns:
            nms_matrix: Matrix where non-maximum pixels are set to zero.
        """
        if self.device == 'cpu':
            # Find local maxima via grayscale dilation
            local_max = cv2.dilate(matrix, self.cv2_kernel)
        else:
            # Find local maxima via 2D max pooling
            local_max = F.max_pool2d(
                matrix, 
                kernel_size=self.ksize, 
                stride=1, 
                padding=self.radio
            )
            
        # Mask out values that are not equal to the local maximum
        nms_matrix = matrix * (matrix == local_max)

        return nms_matrix
    

def get_top_k_array(response_tensor, direction_tensor, k=1000):
    """
    输入: 
        response_tensor: (..., H, W) 任意维度的 Tensor
        direction_tensor: (..., H, W) 形状需与 response 匹配 (可选)
    输出: 
        numpy.ndarray: shape=(k, 4), dtype=float32
        格式: [[x, y, response, direction], ...]
    """
    # 1. 获取维度
    H, W = response_tensor.shape[-2:]
    k = min(k, H * W)

    # 2. 展平 (Flatten)
    # view(-1) 零拷贝，极快
    flat_response = response_tensor.view(-1)

    # 3. TopK (GPU 上极速排序)
    top_vals, top_indices = torch.topk(flat_response, k=k)

    # 4. 计算坐标 (x, y)
    # 注意：在 PyTorch 中，stack 要求所有元素数据类型一致。
    # response 是 float，所以坐标也必须转为 float。
    top_y = top_indices.div(W, rounding_mode='floor').float() # Row
    top_x = (top_indices % W).float()                         # Col

    # 5. 获取 Direction
    if direction_tensor is not None and direction_tensor.numel() > 0:
        # 直接利用 response 的索引去取 direction 的值
        flat_direction = direction_tensor.view(-1)
        top_dirs = flat_direction[top_indices]
    else:
        top_dirs = torch.empty_like(top_vals).fill_(float('nan'))

    # 6. 堆叠 (Stack) -> (k, 4)
    # 注意顺序 [x, y, response, direction]
    result_tensor = torch.stack([top_x, top_y, top_vals, top_dirs], dim=1)

    # 7. 转为 Numpy (单次 I/O)
    return result_tensor.detach().cpu().numpy()


def get_top_k_numpy(response_array, direction_array=None, k=1000):
    """
    输入: 
        response_array: (..., H, W) numpy.ndarray
        direction_array: (..., H, W) 形状需与 response 匹配 (可选)
    输出: 
        numpy.ndarray: shape=(k, 4), dtype=float32
        格式: [[x, y, response, direction], ...]
    """
    # 1. 获取维度
    # 假设输入可能是 (H, W) 或者 (1, 1, H, W)
    shape = response_array.shape
    H, W = shape[-2:]
    
    # 零拷贝展平 (ravel 返回视图，flatten 返回副本，ravel 更快)
    flat_response = response_array.ravel()
    total_pixels = flat_response.size
    k = min(k, total_pixels)

    # 2. TopK 核心优化 (O(N) 复杂度)
    # np.argpartition 不会全排序，只会把最大的 k 个放到数组末尾 (顺序是乱的)
    # 取 -k: 表示取最后 k 个
    unsorted_top_indices = np.argpartition(flat_response, -k)[-k:]
    
    # 3. 对 Top K 进行局部排序 (O(k log k) 复杂度，极快)
    # 因为 argpartition 出来的结果内部是无序的，我们需要按分数降序排列
    unsorted_top_vals = flat_response[unsorted_top_indices]
    
    # argsort 默认升序，[::-1] 翻转为降序
    sort_idx = np.argsort(unsorted_top_vals)[::-1]
    
    # 获取最终排序后的全局索引和分数
    top_indices = unsorted_top_indices[sort_idx]
    top_vals = unsorted_top_vals[sort_idx]

    # 4. 计算坐标 (x, y)
    # np.unravel_index 底层是 C 实现，非常快
    top_y, top_x = np.unravel_index(top_indices, (H, W))

    # 5. 获取 Direction
    if direction_array is not None and direction_array.size > 0:
        # 同样使用 ravel() 来对齐索引
        flat_direction = direction_array.ravel()
        top_dirs = flat_direction[top_indices]
    else:
        # 填充 NaN
        top_dirs = np.full(k, np.nan, dtype=np.float32)

    # 6. 堆叠结果 (预分配内存通常比 stack 快一点点，但 column_stack 也很极致)
    # 为了极致速度，我们创建一个空数组然后填入列
    result = np.empty((k, 4), dtype=np.float32)
    result[:, 0] = top_x  # x
    result[:, 1] = top_y  # y
    result[:, 2] = top_vals # response
    result[:, 3] = top_dirs # direction

    return result


class PostProcessing:
    """
    Post-processing class to apply AreaNMS, get top K, and return list format.
    """

    def __init__(self, device='cpu', nms_radio = 8, get_top_num=1000):
        """
        Args:
            device (str): Computing device ('cpu' or 'cuda').
        """
        self.device = device
        self.area_nms = AreaNMS(radio=nms_radio, device=device)
        self.get_top_num = get_top_num

    def _setup(self):
        self.area_nms._setup()

    def process(self, result):
        """
        Apply AreaNMS to the 'response' key in the result dictionary.

        Args:
            result (dict): Dictionary containing the results with a 'response' key.
        Returns:            result (dict): Updated dictionary with NMS applied to 'response'.
        """
        if 'response' in result:
            response = self.area_nms.process(result['response'])

        if self.device != 'cpu':
            res = get_top_k_array(response, 
                                  result.get('direction', None), 
                                  k=self.get_top_num)
        else:
            res = get_top_k_numpy(response, 
                                  result.get('direction', None), 
                                  k=self.get_top_num)
            
        if res.size == 0:
            res = np.empty((0, 4), dtype=np.float32)
        else:
            max_score = res[0, 2]
            if max_score > 0:
                res[:, 2] /= max_score

        return res
