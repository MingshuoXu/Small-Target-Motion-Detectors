# Update Log

---

## Update Log - Version 2.3.0.7 (Jun 14, 2026)

1. corrected FSTMD default parameters: `Medulla` now explicitly passes `order=12, tau=25` to its `Tm1` and `Mi1` components, and FSTMD applies `n3=5` via `set_para` instead of directly mutating `tm1.order`.

---

## Update Log - Version 2.3.0.6 (Jun 14, 2026)

1. fixed `BaseModel.print_para`: replaced `eval()` with `getattr()` to safely access the private `__paraMappingList` attribute.

2. fixed `BaseModel.set_para`: removed a spurious `self.` prefix in the `getattr` key string (`f'self._…'` → `f'_…'`), which previously caused `set_para` to silently fall back to an empty mapping and skip all parameter updates.

---

## Update Log - Version 2.3.0.5 (Jun 10, 2026)

1. all math operators (`GaussianBlur`, `GammaDelay`, `GammaBandPassFilter`, `SpatialInhibition`) now cache their last forward result in `self.output`, allowing downstream code to read the operator's output without capturing the return value.

2. corrected the package name in `README.md`.

3. optimized startup speed of `main.py` by deferring heavy imports until they are actually needed.

---

## Update Log - Version 2.3.0.4 (Jun 6, 2026)

1. added MPS (Apple Metal Performance Shaders) support in `DeviceSelectorGUI` — new MPS radio button with availability check, and the previous GPU button relabeled to CUDA for clarity.

2. added input validation (`try/except`) in `PostProcessingSelectorGUI` for `show_threshold` and `top_num` fields, so invalid or empty entries fall back to safe defaults instead of raising an unhandled exception.

---

## Update Log - Version 2.3.0.3 (Jun 5, 2026)

1. refactored `ModelAndInputSelectorGUI` into three focused classes: `PostProcessingSelectorGUI`, `DeviceSelectorGUI`, and the new `XTTMP_GUI` — device and post-processor configuration is now selected inside the GUI rather than passed as constructor arguments to `StmdGui`.

2. simplified `FrameVisualizer.update` signature: replaced the `direction` and `process_time` parameters with a single `show_str` for flexible overlay text (e.g. `"cuda : 3.2 ms"`).

3. replaced legacy `compute_temporal_conv`, `compute_circularlist_conv`, and `get_top_k_numpy` with region-proposal helpers: `gen_bboxes_around_points`, `get_STMD_region_proposal`, and `bbox_post_processing`, enabling bounding-box output in addition to dot output.

4. added an early `ImportError` in `main.py` when PyTorch is not installed, with a link to the PyTorch installation guide.

---

## Update Log - - Version 2.3.0 - Torch-First Runtime Refresh (May 30, 2026)

1. removed the numpy-only CPU path and unified inference around torch tensors for CPU, CUDA, and Apple platforms.

2. standardized the model contract around nn.Module with __init__ + forward as the primary interface.

3. kept core lifecycle helpers such as setup and reset_buffer for internal state management and future learning integration.

---

## Update Log - Version 2.2.1 (Sep 25, 2025)

1. use `cv2.dilate` for nms boosting performance in `MatrixNMS` implementation.

2. use `cv2` in visulization to superseed `PIL` for better performance.
