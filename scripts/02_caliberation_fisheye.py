#!/usr/bin/env python3
"""
OpenCV Fisheye Camera Calibration.

This script performs fisheye camera calibration using the equidistant projection
model (OpenCV fisheye module).
"""
import cv2
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.checkerboard_detection import CheckerboardDetector, compute_reprojection_error
from utils.export_formats import (save_opencv_format, save_json_format, 
                                  save_driveworks_compatible, save_calibration_report)


def calibrate_from_video(video_path: Path, 
                        checkerboard_size: tuple,
                        square_size_mm: float,
                        max_frames: int = 50) -> tuple:
    """
    Calibrate camera from video file.
    
    Args:
        video_path: Path to calibration video
        checkerboard_size: (cols, rows) internal corners
        square_size_mm: Size of checkerboard squares in mm
        max_frames: Maximum number of frames to use for calibration
        
    Returns:
        (K, D, rvecs, tvecs, objpoints, imgpoints, image_size)
    """
    print("Extracting calibration frames from video...")
    
    detector = CheckerboardDetector(checkerboard_size, square_size_mm)
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    image_size = (width, height)
    
    # Sample frames evenly throughout video
    frame_indices = np.linspace(0, total_frames - 1, 
                               min(max_frames * 3, total_frames), 
                               dtype=int)
    
    frame_count = 0
    detected_count = 0
    
    pbar = tqdm(total=len(frame_indices), desc="Detecting checkerboards")
    
    for target_frame in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Detect checkerboard
        found, corners = detector.detect(frame, refine_corners=True)
        
        if found:
            objpoints.append(detector.get_object_points())
            imgpoints.append(corners)
            detected_count += 1
            
            if detected_count >= max_frames:
                break
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    print(f"Detected checkerboard in {detected_count}/{frame_count} frames")
    
    if detected_count < 10:
        raise ValueError(f"Not enough detections ({detected_count}). Need at least 10.")
    
    # Perform fisheye calibration
    print("Performing fisheye calibration...")
    
    # Initialize camera matrix
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    
    # Calibration flags for fisheye
    calibration_flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
                        cv2.fisheye.CALIB_CHECK_COND +
                        cv2.fisheye.CALIB_FIX_SKEW)
    
    # Calibration criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    
    # Run calibration
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        image_size,
        K,
        D,
        None,
        None,
        calibration_flags,
        criteria
    )
    
    print(f"RMS reprojection error: {rms:.4f} pixels")
    
    return K, D, rvecs, tvecs, objpoints, imgpoints, image_size


def main():
    # Configuration
    VIDEO_PATH = Path(__file__).parent.parent / "data" / "raw_videos" / "cam0_calib_record2.mp4"
    OUTPUT_DIR = Path(__file__).parent.parent / "data" / "caliberation_results"
    
    # Checkerboard parameters
    CHECKERBOARD_SIZE = (10, 6)  # Internal corners (cols, rows)
    SQUARE_SIZE_MM = 75.0
    
    # Calibration parameters
    MAX_FRAMES = 50  # Use up to 50 frames for calibration
    
    print("=" * 70)
    print("FISHEYE CAMERA CALIBRATION")
    print("=" * 70)
    print(f"Video: {VIDEO_PATH.name}")
    print(f"Checkerboard: {CHECKERBOARD_SIZE[0]}x{CHECKERBOARD_SIZE[1]} internal corners")
    print(f"Square size: {SQUARE_SIZE_MM} mm")
    print()
    
    # Check if video exists
    if not VIDEO_PATH.exists():
        print(f"ERROR: Video file not found: {VIDEO_PATH}")
        return
    
    # Perform calibration
    try:
        K, D, rvecs, tvecs, objpoints, imgpoints, image_size = calibrate_from_video(
            VIDEO_PATH,
            CHECKERBOARD_SIZE,
            SQUARE_SIZE_MM,
            MAX_FRAMES
        )
    except Exception as e:
        print(f"ERROR during calibration: {e}")
        return
    
    # Compute detailed reprojection errors
    print("\nComputing reprojection errors...")
    mean_error, per_view_errors = compute_reprojection_error(
        objpoints, imgpoints, rvecs, tvecs, K, D
    )
    
    print(f"Mean reprojection error: {mean_error:.4f} pixels")
    print(f"Min error: {min(per_view_errors):.4f} pixels")
    print(f"Max error: {max(per_view_errors):.4f} pixels")
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save in multiple formats
    print("Saving calibration results...")
    
    # 1. OpenCV numpy format
    save_opencv_format(K, D, image_size, OUTPUT_DIR)
    
    # 2. JSON format
    save_json_format(
        K, D, image_size,
        OUTPUT_DIR / "calibration.json",
        additional_info={
            "checkerboard_size": CHECKERBOARD_SIZE,
            "square_size_mm": SQUARE_SIZE_MM,
            "num_images": len(objpoints),
            "mean_reprojection_error_pixels": float(mean_error)
        }
    )
    
    # 3. DriveWorks-compatible format
    save_driveworks_compatible(K, D, image_size, OUTPUT_DIR / "driveworks_format.json")
    
    # 4. Human-readable report
    save_calibration_report(
        K, D, image_size,
        mean_error,
        per_view_errors,
        len(objpoints),
        OUTPUT_DIR / "calibration_report.txt"
    )
    
    print()
    print("=" * 70)
    print("CALIBRATION COMPLETE")
    print("=" * 70)
    print(f"Camera Matrix (K):")
    print(K)
    print()
    print(f"Distortion Coefficients (D):")
    print(D.T)
    print()
    print(f"Results saved to: {OUTPUT_DIR}/")
    print()
    
    # Quality assessment
    if mean_error < 0.5:
        quality = "EXCELLENT ✓"
    elif mean_error < 1.0:
        quality = "GOOD ✓"
    elif mean_error < 2.0:
        quality = "ACCEPTABLE"
    else:
        quality = "POOR - Consider recalibration"
    
    print(f"Calibration Quality: {quality}")
    print()


if __name__ == "__main__":
    main()