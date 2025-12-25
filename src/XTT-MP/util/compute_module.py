import numpy as np
import torch

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






