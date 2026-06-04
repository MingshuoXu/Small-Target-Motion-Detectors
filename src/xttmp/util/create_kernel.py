import math

import numpy as np
import torch


def create_2d_gaussian_kernel(size, sigma):
    # 处理参数 (简化防御性编程，假设输入已经是合法的 int 或 tuple)
    sy, sx = (size, size) if isinstance(size, int) else size
    sig_y, sig_x = (sigma, sigma) if isinstance(sigma, (int, float)) else sigma

    # 生成 1D 坐标
    y = torch.arange(sy, dtype=torch.float32) - sy // 2
    x = torch.arange(sx, dtype=torch.float32) - sx // 2
    
    # 生成 2D 网格坐标 (yy 和 xx 都是 size 形状的矩阵)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    # 直接将 2D 坐标代入二维高斯公式（一眼就能看懂的数学表达）
    kernel = torch.exp(-(xx**2 / (2 * sig_x**2) + yy**2 / (2 * sig_y**2)))
    
    # 归一化并调整形状
    kernel /= kernel.sum()
    return kernel.view(1, 1, sy, sx)


def create_gamma_kernel(order, tau, length):
    t = torch.arange(length, dtype=torch.float32)
    
    # 避免 t=0 时 log(0) 报错，加一个极小的值 eps
    t_safe = torch.clamp(t, min=1e-7) 
    
    fact = math.factorial(order - 1)

    # 计算对数域下的结果 (Log Domain)
    # ln(kernel) = order * ln(order * t / tau) - (order * t / tau) - ln(tau * fact)
    log_kernel = (order * torch.log(order * t_safe / tau) 
                  - (order * t_safe / tau) 
                  - math.log(tau * float(fact)))
    
    # 还原回线性域
    kernel = torch.exp(log_kernel)
    
    # 修正 t=0 时的真实值
    kernel[0] = 0.0 
    
    return kernel.view(-1, 1, 1, 1)
        

def create_spatial_inhibition_kernel(kernel_size=15, 
                                sigma1=1.5, 
                                sigma2=3, 
                                e=1., 
                                rho=0, 
                                A=1, 
                                B=3):


    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    y, x = torch.meshgrid(coords, coords, indexing='ij')
    
    # 计算欧式距离矩阵
    r = torch.sqrt(x**2 + y**2)
    
    # 计算双高斯分布 (Difference of Gaussians 的变体)
    # g1 为中心兴奋 (Center)，g2 为周边抑制 (Surround)
    g1 = torch.exp(-(r**2) / (2 * sigma1**2)) / (2 * torch.pi * sigma1**2)
    g2 = torch.exp(-(r**2) / (2 * sigma2**2)) / (2 * torch.pi * sigma2**2)

    g = g1 - e * g2 - rho
    
    kernel = A * g.clamp(min=0) + B * g.clamp(max=0)

    return kernel / kernel.max()


def create_direction_inhi_kernel(KernelSize=8, Sigma1=1.5, Sigma2=3.0):
    """
    Function Description:
    This function generates lateral inhibition kernels along the Theta direction using PyTorch.
    We adopt a one-dimensional DoG as the lateral inhibition kernel function here.
    """
    # 确保输入参数为浮点数张量，以便进行数学运算
    s1 = torch.tensor(Sigma1, dtype=torch.float32)
    s2 = torch.tensor(Sigma2, dtype=torch.float32)

    # Sampling for DoG
    zero_point_factor = (torch.log(s2 / s1) * 2 * (s1**2) * (s2**2)) / (s2**2 - s1**2)
    Zero_Point_DoG_X1 = -torch.sqrt(zero_point_factor)
    Zero_Point_DoG_X2 = -Zero_Point_DoG_X1
    
    Min_Point_DoG_X1 = -torch.sqrt(3 * zero_point_factor)
    Min_Point_DoG_X2 = -Min_Point_DoG_X1

    if KernelSize % 2 == 0:
        KernelSize += 1

    Half_Kernel_Size = (KernelSize - 1) // 2
    Quarter_Kernel_Size = (KernelSize - 1) // 4

    Center_Range_DoG = Zero_Point_DoG_X2 - Zero_Point_DoG_X1
    Center_Step = Center_Range_DoG / Half_Kernel_Size

    Surround_Range_DoG = Min_Point_DoG_X2 - Zero_Point_DoG_X2
    Surround_Step = 2 * Surround_Range_DoG / Quarter_Kernel_Size

    # 使用 torch.arange 替换 np.arange
    # 注意：为了让步长乘法正确应用，arange 生成的序列需要显式指定为 float 类型
    X_Smaller = Zero_Point_DoG_X1 - torch.arange(Quarter_Kernel_Size, 0, -1, dtype=torch.float32) * Surround_Step
    X_Larger = Zero_Point_DoG_X2 + torch.arange(1, Quarter_Kernel_Size + 1, 1, dtype=torch.float32) * Surround_Step
    X_Center = Zero_Point_DoG_X1 + torch.arange(0, Half_Kernel_Size + 1, dtype=torch.float32) * Center_Step
    
    # 使用 torch.cat 替换 np.concatenate
    X = torch.cat((X_Smaller, X_Center, X_Larger))

    # 计算高斯分布
    pi_tensor = torch.tensor(torch.pi)
    Gauss1 = torch.exp(-(X**2) / (2 * s1**2)) / (2 * pi_tensor * s1**2)
    Gauss2 = torch.exp(-(X**2) / (2 * s2**2)) / (2 * pi_tensor * s2**2)
    Inhibition_Kernel = Gauss1 - Gauss2

    # 阈值置零
    Inhibition_Kernel[torch.abs(Inhibition_Kernel) < 1e-4] = 0

    # 使用 torch.reshape 改变形状
    directionalInhiKernel = torch.reshape(Inhibition_Kernel, (1, 1, KernelSize))

    return directionalInhiKernel


def create_T1_kernels(filterNum=4, 
                      alpha=3.0, 
                      eta=1.5, 
                      filterSize=11, 
                      device='cpu'):
    """
    Generate T1 kernels (Difference of Gaussians) using pure PyTorch.
    
    Returns:
        torch.Tensor: A tensor of shape (filterNum, filterSize, filterSize)
    """
    # If the filter size is even, force it to be odd
    if filterSize % 2 == 0:
        filterSize += 1  

    # 1. Compute angles for each filter
    # 形状: (filterNum,) -> 例如 [0, pi/4, pi/2, 3pi/4]
    Theta = torch.arange(filterNum, dtype=torch.float32, device=device) * torch.pi / filterNum

    # 2. Generate coordinates
    r = filterSize // 2
    x_coords = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    y_coords = torch.arange(r, -r - 1, -1, dtype=torch.float32, device=device)
    
    # indexing='xy' 保证了与 numpy.meshgrid 默认行为完全一致
    X, Y = torch.meshgrid(x_coords, y_coords, indexing='xy')

    # 3. 维度扩展以支持批量(向量化)计算
    # 将 X, Y 扩展为 (1, filterSize, filterSize)
    X = X.unsqueeze(0)
    Y = Y.unsqueeze(0)
    
    # 将 Theta 扩展为 (filterNum, 1, 1)
    Theta = Theta.view(-1, 1, 1)

    # 4. 向量化计算偏移量 (X1, Y1, X2, Y2)
    # 利用广播机制，这里会生成形状为 (filterNum, filterSize, filterSize) 的张量
    cos_theta = torch.cos(Theta)
    sin_theta = torch.sin(Theta)

    X1 = X - alpha * cos_theta
    Y1 = Y - alpha * sin_theta
    
    X2 = X + alpha * cos_theta
    Y2 = Y + alpha * sin_theta

    # 5. 生成高斯分布
    coeff = 1 / (2 * torch.pi * eta**2)
    gauss1 = coeff * torch.exp(-(X1**2 + Y1**2) / (2 * eta**2))
    gauss2 = coeff * torch.exp(-(X2**2 + Y2**2) / (2 * eta**2))

    # 6. 计算差值 (最终的滤波器)
    # 形状: (filterNum, filterSize, filterSize)
    dictKernel = gauss1 - gauss2

    # 如果你坚持需要返回 list，可以取消下面这行的注释：
    # return list(dictKernel)
    
    return dictKernel.unsqueeze(1)  # 返回形状为 (filterNum, 1, filterSize, filterSize) 的张量


def create_fracdiff_kernel(alpha=0.8, wide=3):
        """
        Generates a fractional difference kernel.

        Parameters:
        - alpha: The fractional difference parameter.
        - wide: The width of the kernel.

        Returns:
        - frackernel: The fractional difference kernel.
        """
        # Ensure the width is at least 2
        if wide < 2:
            wide = 2
        
        # Initialize the kernel
        frackernel = torch.zeros(wide, dtype=torch.float32)
        
        # Generate the kernel based on alpha
        if alpha == 1:
            frackernel[0] = 1
        elif 0 < alpha < 1:
            t_list = torch.arange(wide)
            frackernel = torch.exp(-alpha * t_list / (1 - alpha)) / (1 - alpha)

            # Normalize the kernel
            sum_kernel = torch.sum(frackernel)  # 1/M(\alpha)
            frackernel = frackernel / sum_kernel
            frackernel[frackernel < 1e-16] = 0
        else:
            raise ValueError("Alpha must be in the interval (0,1].")

        return frackernel


def create_attention_kernel(kernel_size=17, 
                            zeta=[2, 2.5, 3, 3.5], 
                            theta=[0, math.pi/4, math.pi/2, math.pi*3/4],
                            device='cpu',
                            dtype=torch.float32):
    """
    Creates attention kernels using PyTorch.

    Parameters:
    - kernel_size: Size of the kernel.
    - zeta: List of zeta values.
    - theta: List of theta values.
    - device: Target device for the tensors (e.g., 'cpu', 'cuda').
    - dtype: Data type for the tensors.

    Returns:
    - attention_kernel: 2D list containing attention kernels as PyTorch tensors.
    """

    # 调整 kernel size（如果是偶数则加 1）
    if kernel_size % 2 == 0:
        kernel_size += 1

    # 初始化存储 attention kernels 的二维列表
    r = len(zeta)
    s = len(theta)
    attention_kernel = [[None] * s for _ in range(r)]

    # 计算 kernel 中心点
    center = (kernel_size - 1) / 2

    # 生成网格坐标 (注意 dtype 和 device 的传递)
    x = torch.arange(kernel_size, dtype=dtype, device=device) - center
    y = torch.arange(kernel_size, 0, -1, dtype=dtype, device=device) - center
    
    # PyTorch 的 meshgrid 需要明确指定 indexing='xy' 以对齐 NumPy 的默认行为
    shift_x, shift_y = torch.meshgrid(x, y, indexing='xy')

    # 为每种 Zeta 和 Theta 的组合生成 attention kernels
    for i in range(r):
        for j in range(s):
            # 将标量转换为 float 进行计算
            z_val = float(zeta[i])
            t_val = float(theta[j])
            
            # 预计算三角函数以提升效率
            cos_val = math.cos(t_val + math.pi / 2)
            sin_val = math.sin(t_val + math.pi / 2)
            
            # 生成核公式
            term1 = 2 / math.pi / (z_val ** 4)
            term2 = z_val ** 2 - (shift_x * cos_val + shift_y * sin_val) ** 2
            term3 = torch.exp(-(shift_x ** 2 + shift_y ** 2) / (2 * z_val ** 2))
            
            attention_kernel_with_ij = term1 * term2 * term3

            # 阈值过滤 (使用 torch.abs)
            attention_kernel_with_ij[torch.abs(attention_kernel_with_ij) < 1e-4] = 0
            
            # 翻转卷积核 (对应 NumPy 中的 axis=0 和 axis=1)
            attention_kernel_with_ij = torch.flip(attention_kernel_with_ij, dims=[0, 1])
            
            attention_kernel[i][j] = attention_kernel_with_ij

    return attention_kernel


def create_prediction_kernel(Vel=0.25, 
                             Delta_t=25, 
                             filter_size=25, 
                             FilterNum=8, 
                             zeta=2, 
                             eta=2.5,
                             device='cpu',
                             dtype=torch.float32):
    """
    Creates prediction kernels for motion detection using PyTorch.

    Parameters:
    - Vel: Velocity of the moving object.
    - Delta_t: Time interval.
    - filter_size: Size of the filter.
    - FilterNum: Number of filters.
    - zeta: Zeta parameter.
    - eta: Eta parameter.
    - device: Target device for the tensors (e.g., 'cpu', 'cuda').
    - dtype: Data type for the tensors.

    Returns:
    - PredictionKernal: List containing prediction kernels as PyTorch tensors.
    """

    # 初始化存储 prediction kernels 的列表
    PredictionKernal = []

    # 计算中心点
    Center = (filter_size - 1) / 2

    # 生成网格坐标 (与 NumPy 的 meshgrid 对齐)
    x = torch.arange(filter_size, dtype=dtype, device=device) - Center
    y = torch.arange(filter_size, 0, -1, dtype=dtype, device=device) - Center
    ShiftX, ShiftY = torch.meshgrid(x, y, indexing='xy')

    # 计算角度 (注意 torch.atan2 的参数顺序与 np.arctan2 一致，都是 y, x)
    fai = torch.atan2(ShiftY, ShiftX)

    # 计算 Delta X 和 Delta Y
    Delta_X = Vel * Delta_t * torch.cos(fai)
    Delta_Y = Vel * Delta_t * torch.sin(fai)

    # 生成每个方向的 prediction kernels
    for idx in range(FilterNum):
        # 严格遵循原代码逻辑：(idx - 1)
        theta = (idx - 1) * 2 * torch.pi / FilterNum

        # 计算指数项
        PredictionKernalWithIdx = torch.exp(
            -((ShiftX - Delta_X) ** 2 + (ShiftY - Delta_Y) ** 2) / (2 * zeta ** 2)
            + eta * torch.cos(fai - theta)
        )

        # 第一次归一化
        PredictionKernalWithIdx = PredictionKernalWithIdx / torch.sum(PredictionKernalWithIdx)

        # 阈值过滤以加速计算
        PredictionKernalWithIdx[PredictionKernalWithIdx < 5e-4] = 0
        
        # 第二次归一化
        PredictionKernalWithIdx /= torch.sum(PredictionKernalWithIdx)

        # 翻转卷积核 (对应 np.flip axis=0 和 axis=1)
        PredictionKernalWithIdx = torch.flip(PredictionKernalWithIdx, dims=[0, 1])
    
        PredictionKernal.append(PredictionKernalWithIdx)

    return PredictionKernal

