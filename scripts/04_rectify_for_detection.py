#!/usr/bin/env python3
"""
Rectification for 3D Object Detection.

This script prepares rectified images specifically optimized for 3D object detection.
It provides options for different rectification strategies and exports camera parameters
in formats compatible with 3D detection models.
"""
import cv2
import numpy as np
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.export_formats import load_opencv_calibration


def create_rectification_for_detection(K: np.ndarray, 
                                      D: np.ndarray,
                                      image_size: tuple,
                                      balance: float = 0.0,
                                      fov_scale: float = 1.0) -> tuple:
    """
    Create rectification maps optimized for 3D detection.
    
    Args:
        K: Original camera intrinsic matrix
        D: Distortion coefficients
        image_size: (width, height)
        balance: Balance parameter (0.0-1.0)
        fov_scale: Scale factor for FOV (< 1.0 crops, > 1.0 expands)
        
    Returns:
        (map1, map2, new_K, roi) tuple
    """
    # Estimate optimal new camera matrix
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, image_size, np.eye(3), balance=balance, fov_scale=fov_scale
    )
    
    # Create undistortion and rectification maps
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, image_size, cv2.CV_16SC2
    )
    
    # Compute region of interest (valid pixels after undistortion)
    # For fisheye, this is approximate
    roi = (0, 0, image_size[0], image_size[1])
    
    return map1, map2, new_K, roi


def export_detection_calibration(new_K: np.ndarray,
                                 image_size: tuple,
                                 output_path: Path,
                                 original_K: np.ndarray = None) -> None:
    """
    Export rectified camera parameters for 3D detection models.
    
    Many 3D detection models (like SMOKE, MonoDETR, etc.) expect pinhole
    camera parameters after rectification.
    
    Args:
        new_K: Rectified camera intrinsic matrix
        image_size: (width, height)
        output_path: Output JSON path
        original_K: Original (pre-rectification) camera matrix
    """
    calibration = {
        "rectified_intrinsics": {
            "fx": float(new_K[0, 0]),
            "fy": float(new_K[1, 1]),
            "cx": float(new_K[0, 2]),
            "cy": float(new_K[1, 2]),
            "matrix": new_K.tolist()
        },
        "image_size": {
            "width": int(image_size[0]),
            "height": int(image_size[1])
        },
        "projection_matrix": {
            "P": np.hstack([new_K, np.zeros((3, 1))]).tolist(),
            "note": "3x4 projection matrix for 3D detection (P = K * [R|t])"
        },
        "camera_model": "pinhole_rectified",
        "note": "Rectified from fisheye. Distortion removed, can use standard pinhole projection."
    }
    
    if original_K is not None:
        calibration["original_fisheye_intrinsics"] = {
            "fx": float(original_K[0, 0]),
            "fy": float(original_K[1, 1]),
            "cx": float(original_K[0, 2]),
            "cy": float(original_K[1, 2]),
            "matrix": original_K.tolist()
        }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(calibration, f, indent=2)
    
    print(f"Saved detection calibration to: {output_path}")


def rectify_video_for_detection(video_path: Path,
                                output_path: Path,
                                map1: np.ndarray,
                                map2: np.ndarray,
                                image_size: tuple,
                                crop_roi: tuple = None) -> None:
    """
    Rectify video optimized for 3D detection.
    
    Args:
        video_path: Input video
        output_path: Output rectified video
        map1, map2: Rectification maps
        image_size: Output size
        crop_roi: Optional (x, y, w, h) to crop to valid region
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine output size
    out_size = image_size
    if crop_roi is not None:
        x, y, w, h = crop_roi
        out_size = (w, h)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, out_size)
    
    print(f"Rectifying video for detection...")
    print(f"  Input: {video_path.name}")
    print(f"  Output: {output_path.name}")
    print(f"  Output size: {out_size[0]}x{out_size[1]}")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply rectification
        rectified = cv2.remap(frame, map1, map2,
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
        
        # Crop to ROI if specified
        if crop_roi is not None:
            x, y, w, h = crop_roi
            rectified = rectified[y:y+h, x:x+w]
        
        out.write(rectified)
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames", end='\r')
    
    print(f"\n  Completed: {frame_count} frames processed")
    
    cap.release()
    out.release()


def main():
    # Configuration
    VIDEO_PATH = Path(__file__).parent.parent / "data" / "raw_videos" / "cam0_calib_record2.mp4"
    CALIB_DIR = Path(__file__).parent.parent / "data" / "caliberation_results"
    OUTPUT_VIDEO = Path(__file__).parent.parent / "data" / "rectified_for_detection.mp4"
    OUTPUT_CALIB = Path(__file__).parent.parent / "data" / "caliberation_results" / "rectified_camera_params.json"
    SAMPLE_DIR = Path(__file__).parent.parent / "data" / "sample_rectified"
    
    # Rectification parameters
    BALANCE = 0.0      # Minimize blank areas
    FOV_SCALE = 1.0    # Keep original FOV
    
    print("=" * 70)
    print("RECTIFICATION FOR 3D OBJECT DETECTION")
    print("=" * 70)
    print()
    
    # Load calibration
    if not (CALIB_DIR / "camera_matrix.npy").exists():
        print("ERROR: Calibration not found!")
        print("Please run 02_caliberation_fisheye.py first")
        return
    
    print("Loading fisheye calibration...")
    K, D, image_size = load_opencv_calibration(CALIB_DIR)
    print(f"Loaded calibration for {image_size[0]}x{image_size[1]} images")
    print()
    
    # Create rectification maps
    print("Creating rectification maps for 3D detection...")
    print(f"  Balance: {BALANCE}")
    print(f"  FOV Scale: {FOV_SCALE}")
    
    map1, map2, new_K, roi = create_rectification_for_detection(
        K, D, image_size, balance=BALANCE, fov_scale=FOV_SCALE
    )
    
    print()
    print("Rectified Camera Parameters:")
    print(f"  fx: {new_K[0, 0]:.2f} pixels")
    print(f"  fy: {new_K[1, 1]:.2f} pixels")
    print(f"  cx: {new_K[0, 2]:.2f} pixels")
    print(f"  cy: {new_K[1, 2]:.2f} pixels")
    print()
    
    # Export rectified camera parameters for detection models
    export_detection_calibration(new_K, image_size, OUTPUT_CALIB, original_K=K)
    
    # Create sample rectified images
    print("\nCreating sample rectified images...")
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    sample_frames = [0, 500, 1000, 1500, 2000]
    
    for i, frame_idx in enumerate(sample_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        rectified = cv2.remap(frame, map1, map2,
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
        
        # Save comparison
        comparison = np.hstack([frame, rectified])
        cv2.imwrite(str(SAMPLE_DIR / f"sample_{i:02d}_comparison.jpg"), comparison)
        cv2.imwrite(str(SAMPLE_DIR / f"sample_{i:02d}_rectified.jpg"), rectified)
    
    cap.release()
    print(f"Sample images saved to: {SAMPLE_DIR}/")
    
    # Ask about full video rectification
    print()
    print("Do you want to rectify the full video for detection?")
    print("This creates a video ready for 3D object detection inference.")
    response = input("Continue? (y/n): ").strip().lower()
    
    if response != 'y':
        print("\nSkipping full video rectification.")
        print(f"Rectified camera parameters saved to: {OUTPUT_CALIB}")
        print("You can use these parameters with your 3D detection model.")
        return
    
    # Rectify full video
    try:
        rectify_video_for_detection(VIDEO_PATH, OUTPUT_VIDEO, map1, map2, image_size)
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    print()
    print("=" * 70)
    print("RECTIFICATION COMPLETE")
    print("=" * 70)
    print(f"Rectified video: {OUTPUT_VIDEO}")
    print(f"Camera parameters: {OUTPUT_CALIB}")
    print(f"Sample images: {SAMPLE_DIR}/")
    print()
    print("NEXT STEPS FOR 3D DETECTION:")
    print("1. Use rectified video for inference")
    print("2. Load rectified camera parameters in your detection model")
    print("3. Detection results will be in rectified camera coordinate frame")
    print()


if __name__ == "__main__":
    main()