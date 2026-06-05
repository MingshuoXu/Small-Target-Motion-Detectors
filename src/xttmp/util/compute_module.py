from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.nn.functional as F


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
    

class PostProcessing:
    """
    Post-processing class to apply AreaNMS, get top K, and return list format.
    """

    def __init__(self, nms_radio = 8, get_top_num=1000):
        """
        Args:
            nms_radio (int): Radius for AreaNMS.
            get_top_num (int): Number of top points to extract.
        """
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

        res, _ = get_top_k_torch(nms_response, 
                                direction, 
                                k=self.get_top_num)
        if res.shape[0] == 0:
            res = torch.empty((0, 4), device=response.device)
        else:
            max_score = deepcopy(res[0, 2])
            if max_score > 0:
                res[:, 2] /= max_score

        return res


@torch.no_grad()
def gen_bboxes_around_points(results, box_size=16, shift_ratio=0.3):
    """Generate initial bboxes around detected motion points.
    
    Args:
        results: (N, 4) -> [x, y, response, direction]
        box_size: int, box size
        
    Returns:
        [[x1, y1, x2, y2], ... ] (N, 4) tensors
    """
    N = results.shape[0]

    if N == 0:
        return torch.empty((0, 4), device=results.device, dtype=torch.int)
    
    rear_x, rear_y, direction = results[:, 0], results[:, 1], results[:, 3]

    radius = box_size * 0.5
    shift_mag = box_size * shift_ratio

    # 1. 计算偏移量 (dx, dy)，根据方向和预设的 shift_mag
    dx = torch.cos(direction) * shift_mag
    dy = -torch.sin(direction) * shift_mag

    # 2. 一次性将 NaN 偏移量替换为 0.0
    dx = torch.nan_to_num(dx, nan=0.0)
    dy = torch.nan_to_num(dy, nan=0.0)

    # 3. 计算中心点
    center_x = rear_x + dx
    center_y = rear_y + dy

    x1 = (center_x - radius)
    y1 = (center_y - radius)
    x2 = (center_x + radius)
    y2 = (center_y + radius)

    # Stack as (N, 4) -> [x1, y1, x2, y2]
    return torch.stack([x1, y1, x2, y2], dim=1)


@torch.no_grad()
def get_top_k_torch(response_tensor, direction_tensor=None, k=100):
    """
    Extract the top-k points with highest responses from feature maps, filtering out non-positive values.
    
    Args:
        response_tensor (torch.Tensor): The response map tensor of shape (B, 1, H, W) or (B, H, W).
        direction_tensor (torch.Tensor, optional): The corresponding direction map tensor of 
            shape (B, 1, H, W) or (B, H, W). Must match response_tensor's shape. Defaults to None.
        k (int, optional): The maximum number of top points to extract per batch. Defaults to 100.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - results (torch.Tensor): A tensor of shape (M, 4) containing the valid extracted points 
            across the entire batch. M <= B * k. Each row is formatted as [x, y, response, direction].
            - batch_ids (torch.Tensor): A 1D tensor of shape (M,) containing the corresponding 
            batch index (from 0 to B-1) for each point in `results`. dtype is torch.long.
    """
    B, _, H, W = response_tensor.shape
    k = min(k, H * W)
    device = response_tensor.device

    # 1. Flatten -> (B, H*W)
    flat_response = response_tensor.reshape(B, -1)

    # 2. TopK -> top_vals and top_indices are both (B, k)
    top_vals, top_indices = torch.topk(flat_response, k=k, dim=-1)

    # 3. Get Direction -> (B, k)
    if direction_tensor is not None and direction_tensor.numel() > 0:
        flat_direction = direction_tensor.reshape(B, -1)
        top_dirs = torch.gather(flat_direction, dim=-1, index=top_indices)
    else:
        top_dirs = torch.full_like(top_vals, float('nan'))

    # 4. Calculate coordinates (x, y) -> (B, k)
    top_y = top_indices.div(W, rounding_mode='floor').float() 
    top_x = (top_indices % W).float()                        

    # 5. Stack -> merge on the last dimension, shape becomes (B, k, 4)
    stacked = torch.stack([top_x, top_y, top_vals, top_dirs], dim=-1)

    # 6. Generate Mask -> (B, k)
    mask = top_vals > 0

    # 7. Split and filter by Batch
    result_list = []
    batch_id_list = []
    
    for i in range(B):
        batch_mask = mask[i] # Get the mask for the i-th batch
        
        # Apply mask: [k, 4] -> [M_i, 4]
        valid_stacked = stacked[i][batch_mask] 
        result_list.append(valid_stacked)
        
        # Create a batch index tensor of shape (M_i,) filled with the current batch index 'i'
        batch_id_list.append(torch.full((valid_stacked.shape[0],), i, device=device, dtype=torch.long))
    
    # Concatenate all valid items into continuous tensors
    return torch.cat(result_list, dim=0), torch.cat(batch_id_list, dim=0)


@torch.no_grad()
def get_STMD_region_proposal(response_tensor, direction_tensor=None, top_k=1, box_size=16, spatial_scale=1.0, shift_ratio=0.3):
    nms_win = int(box_size * spatial_scale) | 1  # 确保是奇数
    score_mask = F.max_pool2d(response_tensor, kernel_size=nms_win, stride=1, padding=nms_win//2)
    
    nms_response_tensor = torch.where(response_tensor == score_mask, response_tensor, 0.0)

    vSTMD_res, batch_id = get_top_k_torch(nms_response_tensor, direction_tensor, k=top_k)

    if spatial_scale > 1:
        vSTMD_res[:, :2] *= spatial_scale   # 将坐标放大回原图尺度

    bboxes = gen_bboxes_around_points(vSTMD_res, box_size, shift_ratio)

    return vSTMD_res, bboxes, batch_id


@torch.no_grad()
def bbox_post_processing(top_k=1, box_size=16, spatial_scale=1.0, shift_ratio=0.3):

    def post_process_func(
        response_tensor,
        direction_tensor=None
    ):
        vSTMD_res, bboxes, _ =  get_STMD_region_proposal(response_tensor,
                                                        direction_tensor,
                                                        top_k=top_k,
                                                        box_size=box_size,
                                                        spatial_scale=spatial_scale,
                                                        shift_ratio=shift_ratio )
        return torch.cat([bboxes, vSTMD_res[..., 2:3]], dim=1)
    
    return post_process_func

