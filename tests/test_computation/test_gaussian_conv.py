import os
import sys
import torch
import time


import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF

filePath = os.path.realpath(__file__)
project_path = os.path.dirname(os.path.dirname(os.path.dirname(filePath)))
sys.path.append(os.path.join(project_path, 'src'))
from xttmp.util.create_kernel import create_gaussian_kernel

# --- 1. 自定义实现 (模仿 torchvision 的逻辑) ---
def custom_gaussian_blur(img, kernel, kernel_size, sigma):
    """
    为了和 torchvision 对齐，必须：
    1. 生成完全相同的高斯核
    2. 使用 Reflection Padding (反射填充) 处理边缘
    """
    
    pad_size = kernel_size // 2
    img_padded = F.pad(img, (pad_size, pad_size, pad_size, pad_size), mode='reflect')

    output = F.conv2d(img_padded, kernel, groups=img.shape[1])
 
    return output

# --- 2. 测试与评估函数 ---
def test_correctness_and_speed():
    # 设置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用设备: {device}")
    
    B, C, H, W = 4, 3, 512, 512
    k_size = 5
    sigma_val = 1.5
    
    # 生成随机图片
    img = torch.rand(B, C, H, W).to(device)

    kernel = create_gaussian_kernel(k_size, sigma_val)
    kernel = torch.from_numpy(kernel).to(img.device).float().unsqueeze(0).unsqueeze(0)  # 转为张量并移动到正确设备
    kernel = kernel.repeat(img.shape[1], 1, 1, 1)
    
    # --- 运行 Torchvision 版本 ---
    # 注意：torchvision 接收的 sigma 是列表 [sigma, sigma]
    for _ in range(10):
        out_tv = TVF.gaussian_blur(img, kernel_size=[k_size, k_size], sigma=[sigma_val, sigma_val]) # warm up
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_tv = time.time()
    for _ in range(100):
        out_tv = TVF.gaussian_blur(img, kernel_size=[k_size, k_size], sigma=[sigma_val, sigma_val])
    torch.cuda.synchronize() if device.type == 'cuda' else None
    time_tv = time.time() - start_tv
    
    # --- 运行 自定义 版本 ---
    for _ in range(10):
        out_tv = custom_gaussian_blur(img, kernel, k_size, sigma_val)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_custom = time.time()
    for _ in range(100):
        out_custom = custom_gaussian_blur(img, kernel, k_size, sigma_val)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    time_custom = time.time() - start_custom
    
    # --- 比较结果 ---
    # 计算最大绝对误差
    diff = (out_tv - out_custom).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # 使用 torch.allclose 判断是否“相等”
    # atol=1e-5 对于 float32 图像处理是可以接受的误差范围
    is_same = torch.allclose(out_tv, out_custom, atol=1e-5)
    
    print("-" * 40)
    print(f"最大像素差异 (Max Diff): {max_diff:.8f}")
    print(f"平均像素差异 (Mean Diff): {mean_diff:.8f}")
    print(f"结果是否一致 (atol=1e-5): {'✅ 是' if is_same else '❌ 否'}")
    print("-" * 40)
    print(f"Torchvision 耗时: {time_tv*1000:.4f} ms")
    print(f"自定义实现 耗时:   {time_custom*1000:.4f} ms")
    
    if not is_same:
        print("\n[分析] 如果结果不一致，通常是因为：")
        print("1. 边缘填充方式不同 (Reflect vs Zero)。")
        print("2. 浮点数计算顺序导致的微小精度误差。")

if __name__ == '__main__':
    test_correctness_and_speed()