# Small Target Motion Detectors, Version 2.3 (XTT-MP: Extremely Tiny Target - Motion Perception)

[![PyPI version](https://img.shields.io/pypi/v/xttmp.svg)](https://pypi.org/project/xttmp/) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/pytorch-1.10%2B-ee4c2c)](https://pytorch.org/)

> See [update_log.md](update_log.md) for the full version history.

**XTT-MP** (Extremely Tiny Target - Motion Perception) is a natural architecture-based framework specifically designed for detecting and perceiving the motion of **extremely small targets** in complex environments.

This release is torch-first: CPU, CUDA, and Apple platforms all use the same torch tensor path, so there is no longer a numpy-only CPU branch to maintain. The model layer now follows a standard `nn.Module` flow with `__init__` and `forward`, while core blocks keep their internal lifecycle helpers such as `setup` and `reset_buffer`.

Built with modularity and extensibility in mind, XTT-MP provides a robust suite of tools for researchers and developers to iterate on tiny-object detection and motion analysis algorithms.

---

## ✨ Key Features

- **Tiny Target Specialist**: Optimized feature extraction and attention mechanisms tailored for sub-8x8 pixel objects.
- **Motion-Aware Architecture**: Integrated spatiotemporal modules to enhance temporal consistency and motion trajectory estimation.
- **Decoupled Design**: model and core layers are separated cleanly, which makes future learning modules easier to integrate.
- **Torch-Only Runtime**: inference and post-processing stay in torch tensors on both CPU and GPU, with Apple users able to run the same code path without a CUDA-specific fallback.

## 📦 Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3+ (optional, for NVIDIA acceleration)
- Apple Silicon or macOS users can use the same torch-based workflow without numpy-specific CPU code.

### Version Notes
- Model classes now expose the standard `__init__` + `forward` flow.
- Core classes provide lifecycle helpers such as `setup` and `reset_buffer`.
- CPU and GPU outputs are both represented as torch tensors.

### Example Resources
- Example data and demo assets stay in the GitHub repository under `example-data/`.
- They are not packaged into the PyPI distribution.
- After `pip install xttmp`, use the installed code and bring your own input data, or run from a repository checkout to access the bundled examples.

### Via PyPI
#### CPU / MPS
```bash
pip install xttmp[torch]
```

#### NVIDIA GPU (CUDA 12.6)
```bash
pip install xttmp
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Running the GUI Demo
```bash
xttmp_gui
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