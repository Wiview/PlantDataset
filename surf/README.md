# SURF Feature Detection for COLMAP

Simple SURF feature detection and matching pipeline for COLMAP reconstruction.

## Usage

```bash
python colmap_surf.py --proj /path/to/project --images images --camera SIMPLE_RADIAL
```

## Files

- `detector.py` - SURF keypoint detection
- `matchers.py` - Feature matching algorithms  
- `database.py` - COLMAP database interface
- `colmap_surf.py` - Main pipeline

## Requirements

- OpenCV with SURF support
- NumPy
- COLMAP
