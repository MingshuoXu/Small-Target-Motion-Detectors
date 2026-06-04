# Update Log

---
## Update Log - - Version 2.3.0 - Torch-First Runtime Refresh (May 30, 2026)

1. removed the numpy-only CPU path and unified inference around torch tensors for CPU, CUDA, and Apple platforms.

2. standardized the model contract around nn.Module with __init__ + forward as the primary interface.

3. kept core lifecycle helpers such as setup and reset_buffer for internal state management and future learning integration.

---
## Update Log - Version 2.2.1 (Sep 25, 2025)

1. use `cv2.dilate` for nms boosting performance in `MatrixNMS` implementation.

2. use `cv2` in visulization to superseed `PIL` for better performance.
