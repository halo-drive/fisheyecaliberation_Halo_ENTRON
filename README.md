# Fisheye Camera Calibration

Complete OpenCV-based fisheye camera calibration pipeline for 3D object detection applications.

## Overview

This repository provides tools to calibrate fisheye cameras, undistort/rectify images, and export camera parameters in formats compatible with 3D object detection models.

**Key Features:**
- OpenCV fisheye calibration (equidistant projection model)
- Automated checkerboard detection
- Undistortion and rectification
- Multiple export formats (OpenCV, JSON, DriveWorks-compatible)
- Quality analysis and visualization
- Optimized for 3D detection workflows

## Camera Information

- **Camera**: Entron F008A GMSL Fisheye
- **Resolution**: 3848 × 2168 pixels
- **Checkerboard**: 10×6 internal corners (11×7 squares)
- **Square Size**: 75mm

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.7+
- OpenCV 4.5+ (with contrib modules)
- NumPy
- Matplotlib
- tqdm

## Quick Start

### 1. Record Calibration Video

Record a video moving the checkerboard through these positions:
- All 9 regions (top-left, top-center, top-right, middle-left, center, middle-right, bottom-left, bottom-center, bottom-right)
- 3 distances (close, medium, far)
- Some tilted angles

**Requirements:**
- 30-50 good detections minimum
- 100% region coverage
- Hold still for 3-4 seconds at each position

Place video in: `data/raw_videos/cam0_calib_record2.mp4`

### 2. Run Calibration Pipeline

```bash
# Optional: Extract frames (helps verify checkerboard detection)
python scripts/01_extract_frames.py

# Perform fisheye calibration
python scripts/02_caliberation_fisheye.py

# Generate undistorted video
python scripts/03_undistort_video.py

# Create rectified images for 3D detection
python scripts/04_rectify_for_detection.py

# Visualize calibration quality
python scripts/05_visualize_results.py
```

## Directory Structure

```
fisheye-calibration/
├── data/
│   ├── raw_videos/              # Input calibration videos
│   │   └── cam0_calib_record2.mp4
│   ├── caliberation_results/    # Calibration outputs
│   │   ├── camera_matrix.npy
│   │   ├── distortion_coeffs.npy
│   │   ├── calibration.json
│   │   ├── calibration_report.txt
│   │   ├── driveworks_format.json
│   │   ├── rectified_camera_params.json
│   │   └── visualizations/
│   ├── sample_undistorted/      # Sample undistorted images
│   └── extracted_frames/        # Extracted calibration frames
├── scripts/
│   ├── 01_extract_frames.py
│   ├── 02_caliberation_fisheye.py
│   ├── 03_undistort_video.py
│   ├── 04_rectify_for_detection.py
│   └── 05_visualize_results.py
└── utils/
    ├── checkerboard_detection.py
    └── export_formats.py
```

## Scripts Overview

### 01_extract_frames.py
Extracts frames with successful checkerboard detection from video.

**Parameters:**
- `SKIP_FRAMES`: Process every Nth frame (default: 10)
- `BLUR_THRESHOLD`: Reject blurry frames (default: 100.0)
- `MIN/MAX_AREA_RATIO`: Valid checkerboard size range

### 02_caliberation_fisheye.py
Main calibration script using OpenCV fisheye model.

**Features:**
- Automatic checkerboard detection
- Fisheye-specific calibration
- Quality metrics (RMS reprojection error)
- Multiple export formats

**Output Quality:**
- **Excellent**: < 0.5 pixels mean error
- **Good**: < 1.0 pixels
- **Acceptable**: < 2.0 pixels

### 03_undistort_video.py
Removes fisheye distortion from video.

**Parameters:**
- `BALANCE`: 0.0 = minimize blank areas, 1.0 = retain all pixels

**Outputs:**
- Undistorted video
- Sample comparison images

### 04_rectify_for_detection.py
Optimized rectification for 3D object detection.

**Features:**
- Creates pinhole-rectified images
- Exports detection-ready camera parameters
- Removes all distortion

**Outputs:**
- `rectified_camera_params.json`: Use with 3D detection models
- Rectified video (optional)

### 05_visualize_results.py
Generate calibration quality visualizations.

**Outputs:**
- Calibration summary
- Distortion pattern visualization
- Reprojection error analysis

## Output Formats

### 1. OpenCV Format (.npy)
```python
K = np.load('camera_matrix.npy')          # 3×3 intrinsic matrix
D = np.load('distortion_coeffs.npy')      # 4×1 distortion coeffs
image_size = np.load('image_size.npy')    # (width, height)
```

### 2. JSON Format
```json
{
  "camera_matrix": {
    "fx": 1234.56,
    "fy": 1234.56,
    "cx": 1924.0,
    "cy": 1084.0
  },
  "distortion_coefficients": {
    "k1": -0.123,
    "k2": 0.456,
    "k3": -0.789,
    "k4": 0.012
  }
}
```

### 3. Rectified Camera Parameters (for 3D Detection)
```json
{
  "rectified_intrinsics": {
    "fx": 1234.56,
    "fy": 1234.56,
    "cx": 1924.0,
    "cy": 1084.0
  },
  "projection_matrix": {
    "P": [[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]]
  },
  "camera_model": "pinhole_rectified"
}
```

### 4. DriveWorks-Compatible Format
Approximate conversion for use with NVIDIA DriveWorks SDK.

**Note**: OpenCV uses equidistant projection, DriveWorks uses f-theta. For small FOV they're similar, but may need refinement in DriveWorks tools.

## Usage in 3D Detection Pipeline

### Step 1: Calibrate Camera
```bash
python scripts/02_caliberation_fisheye.py
```

### Step 2: Rectify Images
```bash
python scripts/04_rectify_for_detection.py
```

### Step 3: Use in Detection Model
```python
import json
import numpy as np

# Load rectified camera parameters
with open('data/caliberation_results/rectified_camera_params.json') as f:
    cam_params = json.load(f)

# Get intrinsics for your model
fx = cam_params['rectified_intrinsics']['fx']
fy = cam_params['rectified_intrinsics']['fy']
cx = cam_params['rectified_intrinsics']['cx']
cy = cam_params['rectified_intrinsics']['cy']

# Projection matrix (3×4)
P = np.array(cam_params['projection_matrix']['P'])

# Use in your 3D detection model
# Note: Images must be rectified using script 04
```

## Troubleshooting

### Low Detection Rate
- **Check lighting**: Ensure even lighting, no glare on checkerboard
- **Verify focus**: Camera must be in focus
- **Adjust thresholds**: Lower `BLUR_THRESHOLD` in script 01
- **Check checkerboard size**: Verify 10×6 internal corners

### High Reprojection Error (>1.0 px)
- **More images**: Capture 50-100 frames instead of 30
- **Better coverage**: Ensure all 9 regions covered at 3 distances
- **Check board flatness**: Warped checkerboards cause errors
- **Verify square size**: Confirm 75mm measurement is accurate

### Blank Areas After Undistortion
- **Adjust balance**: Increase `BALANCE` parameter (0.0 → 0.5)
- **Crop images**: Use ROI from rectification
- **Accept some loss**: Fisheye edges always have some blank areas

### DriveWorks Compatibility Issues
- **Use as starting point**: DriveWorks format is approximate
- **Refine in DriveWorks**: Use their calibration tools for final tuning
- **Consider native workflow**: For production, use DriveWorks end-to-end

## Camera Model Details

**OpenCV Fisheye Model (Equidistant Projection):**
```
x_distorted = f * θ * (x_normalized / r)
y_distorted = f * θ * (y_normalized / r)

where:
  θ = atan(r)  (angle from optical axis)
  r = sqrt(x_normalized² + y_normalized²)
  
Distortion polynomial:
  θ_d = θ(1 + k1*θ² + k2*θ⁴ + k3*θ⁶ + k4*θ⁸)
```

**Parameters:**
- `K`: Intrinsic matrix [fx, 0, cx; 0, fy, cy; 0, 0, 1]
- `D`: Distortion coefficients [k1, k2, k3, k4]

## References

- [OpenCV Fisheye Calibration](https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html)
- [NVIDIA DriveWorks Documentation](https://developer.nvidia.com/drive/drive-sdk)
- [Camera Calibration Theory](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

## License

Internal use - Halo Drive Ltd

## Author

Ash - Founding Software Engineer, Halo Drive LTD