from copy import deepcopy

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


def compute_response(ipt):
    """
    Computes the maximum response from multiple inputs.

    Parameters:
    - ipt: List containing input data.

    Returns:
    - response: Maximum response computed from the inputs.
    """
    return torch.amax(ipt, dim=1, keepdim=True)


def compute_direction(ipt):
    """
    Compute the dominant direction given a set of directional responses

    Parameters:
    - ipt: List containing directional responses.

    Returns:
    - direction_opt: Dominant direction computed from the responses.
    """

    B, C, H, W = ipt.shape # C = 8 (numDirection)
    device = ipt.device  # 获取输入信号所在的设备

    # 1. 预计算每个通道对应的单位向量角度 (theta)
    # angles = [0, 1/8*2pi, 2/8*2pi, ...]
    angles = torch.linspace(0, 2 * torch.pi, steps=C+1, device=device)[:-1]
    
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
    # 只有当两个分量都非常小时才设为 NaN
    mask = (output_sin == 0) & (output_cos == 0)
    direction_opt[mask] = float('nan')
    direction_opt = direction_opt.unsqueeze(0)  # 去掉批次维度
    
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

    def __init__(self, radio=8):
        """
        Args:
            radio (int): Neighborhood radius. Window size = 2 * radio + 1.
            device (str): Computing device ('cpu' or 'cuda').
        """
        self.radio = radio
        self.ksize = self.radio * 2 + 1

    def __call__(self, matrix):
        """
        Apply NMS to the input matrix.

        Args:
            matrix (torch.Tensor): Input heatmap or score map.
                - If CUDA: Expected shape (B, C, H, W) as torch tensor.

        Returns:
            nms_matrix: Matrix where non-maximum pixels are set to zero.
        """
        # Find local maxima via 2D max pooling
        local_max = F.max_pool2d(
            matrix, 
            kernel_size=self.ksize, 
            stride=1, 
            padding=self.radio
        )

        return matrix * (matrix == local_max)
    

def get_top_k_torch(response_tensor, direction_tensor, k=1000):
    """
    输入: 
        response_tensor: (..., H, W) 任意维度的 Tensor
        direction_tensor: (..., H, W) 形状需与 response 匹配 (可选)
    输出: 
        torch.Tensor: shape=(M, 4), dtype=float32, 其中 M <= k
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

    # 4. 过滤掉 <= 0 的值 ---
    # 创建掩码：只保留大于 0 的值
    mask = top_vals > 0
    
    # 如果全都是 0，直接返回空数组，避免后续报错
    if not mask.any():
        return torch.empty((0, 4))

    # 应用掩码，缩减 tensor 长度
    top_vals = top_vals[mask]
    top_indices = top_indices[mask]
    # ------------------------------------

    # 5. 计算坐标 (x, y)
    # 此时计算量已经减少，只计算非零点
    top_y = top_indices.div(W, rounding_mode='floor').float() 
    top_x = (top_indices % W).float()                        

    # 6. 获取 Direction
    if direction_tensor is not None and direction_tensor.numel() > 0:
        flat_direction = direction_tensor.view(-1)
        # 注意：这里使用过滤后的 top_indices
        top_dirs = flat_direction[top_indices]
    else:
        top_dirs = torch.empty_like(top_vals).fill_(float('nan'))

    # 7. 堆叠 (Stack) -> (M, 4)
    result_tensor = torch.stack([top_x, top_y, top_vals, top_dirs], dim=1)

    return result_tensor


def get_top_k_numpy(response_array, direction_array=None, k=1000):
    """
    输入: 
        response_array: (..., H, W) numpy.ndarray
        direction_array: (..., H, W) (可选)
    输出: 
        numpy.ndarray: shape=(M, 4), dtype=float32, 其中 M <= k
        格式: [[x, y, response, direction], ...]
    """
    # 1. 获取维度
    shape = response_array.shape
    H, W = shape[-2:]
    
    # 零拷贝展平
    flat_response = response_array.ravel()
    k = min(k, flat_response.size)

    # 2. TopK 核心优化 (O(N))
    # argpartition 找出最大的 k 个 (无序)
    unsorted_top_indices = np.argpartition(flat_response, -k)[-k:]
    unsorted_top_vals = flat_response[unsorted_top_indices]
    
    # 3. 局部排序 (O(k log k))
    # argsort 默认升序，[::-1] 翻转为降序
    sort_idx = np.argsort(unsorted_top_vals)[::-1]
    
    # 获取排序后的 Top K 索引和值
    top_indices = unsorted_top_indices[sort_idx]
    top_vals = unsorted_top_vals[sort_idx]

    # --- [关键修改] 4. 过滤掉 <= 0 的值 ---
    # 创建掩码
    mask = top_vals > 0
    
    # 极速判断：如果没有有效值，直接返回空数组
    # np.any() 很快
    if not np.any(mask):
        return np.empty((0, 4), dtype=np.float32)
        
    # 应用掩码 (切片操作，只保留有效值)
    # 因为 k 通常不大 (比如 1000)，这里的拷贝开销可忽略不计
    top_vals = top_vals[mask]
    top_indices = top_indices[mask]
    
    # 更新实际数量 M
    M = top_vals.size
    # ------------------------------------

    # 5. 计算坐标 (x, y)
    # 只对过滤后的索引计算，节省算力
    top_y, top_x = np.unravel_index(top_indices, (H, W))

    # 6. 获取 Direction
    if direction_array is not None and direction_array.size > 0:
        flat_direction = direction_array.ravel()
        top_dirs = flat_direction[top_indices]
    else:
        top_dirs = np.full(M, np.nan, dtype=np.float32)

    # 7. 堆叠结果
    # 分配恰好大小为 M 的内存
    result = np.empty((M, 4), dtype=np.float32)
    result[:, 0] = top_x       # x
    result[:, 1] = top_y       # y
    result[:, 2] = top_vals    # response
    result[:, 3] = top_dirs    # direction

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
        self.area_nms = AreaNMS(radio=nms_radio)
        self.get_top_num = get_top_num

    def __call__(self, response, direction=None):
        if self.get_top_num == 1:
            idx = torch.argmax(response)
            y, x = divmod(idx.item(), response.shape[-1])
            response_value = response[0, 0, y, x].item()
            direction_value = direction[0, 0, y, x].item() if direction is not None else float('nan')
            return torch.tensor([[x, y, response_value, direction_value]])
        else:
            return self.process(response, direction)

    def process(self, response, direction=None):
        """
        Apply AreaNMS to the 'response' key in the result dictionary.

        Args:
            result (dict): Dictionary containing the results with a 'response' key.
        Returns:            result (dict): Updated dictionary with NMS applied to 'response'.
        """
        nms_response = self.area_nms(response)

        res = get_top_k_torch(nms_response, 
                                direction, 
                                k=self.get_top_num)
        if res.shape[0] == 0:
            res = torch.empty((0, 4), device=response.device)
        else:
            max_score = deepcopy(res[0, 2])
            if max_score > 0:
                res[:, 2] /= max_score

        return res
