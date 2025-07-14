# SuperPoint Feature Detection for COLMAP

PyTorch implementation of SuperPoint feature detection for COLMAP reconstruction.

## Usage

```bash
python colmap_superpoint.py --proj /path/to/project --images images --max_kpts 1000
```

## Files

- `detector.py` - SuperPoint neural network
- `extract.py` - Feature extraction wrapper
- `matchers.py` - Feature matching algorithms
- `colmap_superpoint.py` - Main pipeline

## Requirements

- PyTorch
- OpenCV
- NumPy
- COLMAP
- SuperPoint model weights (superpoint_v1.pth)
