# XTT-MP: Extremely Tiny Target - Motion Perception

[![PyPI version](https://img.shields.io/pypi/v/XTT-MP.svg)](https://pypi.org/project/XTT-MP/) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/pytorch-1.10%2B-ee4c2c)](https://pytorch.org/)


**XTT-MP** (Extremely Tiny Target - Motion Perception) is a natural architecture-based framework specifically designed for detecting and perceiving the motion of **extremely small targets** in complex environments.

Built with modularity and extensibility in mind, XTT-MP provides a robust suite of tools for researchers and developers to iterate on tiny-object detection and motion analysis algorithms.

---

## ✨ Key Features

- **Tiny Target Specialist**: Optimized feature extraction and attention mechanisms tailored for sub-8x8 pixel objects.
- **Motion-Aware Architecture**: Integrated spatiotemporal modules to enhance temporal consistency and motion trajectory estimation.
- **Decoupled Design**: 
- **High Performance**: Optimized CUDA kernels and data pipelines for efficient training and inference.

## 📦 Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3+ (recommended)

### Via PyPI
```bash
pip install XTT-MP
```

### Citation

If you find this project useful for your research, please consider citing by this.
```
@misc{STMDgit,
	author       = {Xu, Mingshuo},
	title        = {Small-Target-Motion-Detectors, Version 2},
	year         = {2024},
	url          = {https://github.com/MingshuoXu/Small-Target-Motion-Detectors},
	note         = {Accessed: 2025-12-12}
}
```