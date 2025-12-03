#!/usr/bin/env python3
"""
Undistort video using fisheye calibration results.

This script applies the calibration to remove fisheye distortion from video.
"""
import cv2
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.export_formats import load_opencv_calibration


def create_undistortion_maps(K: np.ndarray, D: np.ndarray, 
                            image_size: tuple, 
                            balance: float = 0.0) -> tuple:
    """
    Create maps for undistorting fisheye images.
    
    Args:
        K: Camera intrinsic matrix
        D: Distortion coefficients
        image_size: (width, height) of images
        balance: Balance between retaining all pixels (1.0) and minimizing 
                blank areas (0.0)
    
    Returns:
        (map1, map2, new_K) for use with cv2.remap
    """
    # Compute optimal new camera matrix
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, image_size, np.eye(3), balance=balance
    )
    
    # Generate undistortion maps
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, image_size, cv2.CV_16SC2
    )
    
    return map1, map2, new_K


def undistort_video(video_path: Path, 
                   output_path: Path,
                   K: np.ndarray, 
                   D: np.ndarray,
                   image_size: tuple,
                   balance: float = 0.0) -> None:
    """
    Undistort entire video.
    
    Args:
        video_path: Input video path
        output_path: Output video path
        K: Camera intrinsic matrix
        D: Distortion coefficients
        image_size: (width, height)
        balance: Undistortion balance parameter
    """
    print(f"Creating undistortion maps (balance={balance})...")
    map1, map2, new_K = create_undistortion_maps(K, D, image_size, balance)
    
    print(f"Opening video: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height} @ {fps:.2f} fps, {total_frames} frames")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, image_size)
    
    if not out.isOpened():
        raise ValueError(f"Cannot create output video: {output_path}")
    
    print("Undistorting video...")
    pbar = tqdm(total=total_frames, desc="Processing frames")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Undistort frame
        undistorted = cv2.remap(frame, map1, map2, 
                               interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT)
        
        out.write(undistorted)
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    print(f"Processed {frame_count} frames")
    print(f"Undistorted video saved to: {output_path}")


def undistort_sample_images(video_path: Path,
                           output_dir: Path,
                           K: np.ndarray,
                           D: np.ndarray,
                           image_size: tuple,
                           num_samples: int = 5,
                           balance: float = 0.0) -> None:
    """
    Extract and undistort sample images from video for visual inspection.
    
    Args:
        video_path: Input video path
        output_dir: Output directory for sample images
        K: Camera intrinsic matrix
        D: Distortion coefficients
        image_size: (width, height)
        num_samples: Number of sample images to extract
        balance: Undistortion balance parameter
    """
    print(f"\nCreating {num_samples} sample undistorted images...")
    
    map1, map2, new_K = create_undistortion_maps(K, D, image_size, balance)
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames evenly throughout video
    sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, frame_idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Undistort
        undistorted = cv2.remap(frame, map1, map2,
                               interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT)
        
        # Create side-by-side comparison
        comparison = np.hstack([frame, undistorted])
        
        # Save both
        cv2.imwrite(str(output_dir / f"sample_{i:02d}_original.jpg"), frame)
        cv2.imwrite(str(output_dir / f"sample_{i:02d}_undistorted.jpg"), undistorted)
        cv2.imwrite(str(output_dir / f"sample_{i:02d}_comparison.jpg"), comparison)
    
    cap.release()
    print(f"Sample images saved to: {output_dir}/")


def main():
    # Configuration
    VIDEO_PATH = Path(__file__).parent.parent / "data" / "raw_videos" / "cam0_calib_record2.mp4"
    CALIB_DIR = Path(__file__).parent.parent / "data" / "caliberation_results"
    OUTPUT_VIDEO = Path(__file__).parent.parent / "data" / "undistorted_video.mp4"
    SAMPLE_DIR = Path(__file__).parent.parent / "data" / "sample_undistorted"
    
    # Undistortion parameters
    BALANCE = 0.0  # 0.0 = minimize blank areas, 1.0 = retain all pixels
    
    print("=" * 70)
    print("VIDEO UNDISTORTION")
    print("=" * 70)
    print(f"Input video: {VIDEO_PATH.name}")
    print(f"Balance: {BALANCE}")
    print()
    
    # Check if calibration exists
    if not (CALIB_DIR / "camera_matrix.npy").exists():
        print("ERROR: Calibration not found!")
        print(f"Please run 02_caliberation_fisheye.py first")
        return
    
    # Load calibration
    print("Loading calibration...")
    try:
        K, D, image_size = load_opencv_calibration(CALIB_DIR)
        print(f"Loaded calibration for {image_size[0]}x{image_size[1]} images")
    except Exception as e:
        print(f"ERROR loading calibration: {e}")
        return
    
    # Check if video exists
    if not VIDEO_PATH.exists():
        print(f"ERROR: Video file not found: {VIDEO_PATH}")
        return
    
    # Create sample undistorted images first
    try:
        undistort_sample_images(VIDEO_PATH, SAMPLE_DIR, K, D, image_size,
                               num_samples=5, balance=BALANCE)
    except Exception as e:
        print(f"ERROR creating sample images: {e}")
    
    # Ask user if they want to undistort full video
    print()
    print("Do you want to undistort the full video? This may take several minutes.")
    response = input("Continue? (y/n): ").strip().lower()
    
    if response != 'y':
        print("Skipping full video undistortion.")
        print("Sample images are available in:", SAMPLE_DIR)
        return
    
    # Undistort full video
    try:
        undistort_video(VIDEO_PATH, OUTPUT_VIDEO, K, D, image_size, balance=BALANCE)
    except Exception as e:
        print(f"ERROR undistorting video: {e}")
        return
    
    print()
    print("=" * 70)
    print("UNDISTORTION COMPLETE")
    print("=" * 70)
    print(f"Undistorted video: {OUTPUT_VIDEO}")
    print(f"Sample images: {SAMPLE_DIR}/")
    print()


if __name__ == "__main__":
    main()