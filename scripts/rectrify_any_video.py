#!/usr/bin/env python3
"""
Standalone Video Rectification Script

Rectify ANY fisheye video using saved calibration results.
Use this to pre-process videos before running 3D detection inference.

Usage:
    python rectify_any_video.py --input raw_video.mp4 --output rectified_video.mp4 --calib-dir ./data/caliberation_results
"""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def load_calibration(calib_dir):
    """Load fisheye calibration from directory."""
    calib_dir = Path(calib_dir)
    
    # Load camera matrix and distortion coefficients
    K = np.load(calib_dir / 'camera_matrix.npy')
    D = np.load(calib_dir / 'distortion_coeffs.npy')
    image_size = tuple(np.load(calib_dir / 'image_size.npy'))
    
    return K, D, image_size


def create_rectification_maps(K, D, image_size, balance=0.0, fov_scale=1.0):
    """
    Create rectification maps for fisheye undistortion.
    
    Args:
        K: Camera intrinsic matrix (3x3)
        D: Distortion coefficients (4x1)
        image_size: (width, height)
        balance: 0.0 = minimize blank areas, 1.0 = retain all pixels
        fov_scale: Field of view scale factor
    
    Returns:
        (map1, map2, new_K) for cv2.remap
    """
    # Estimate optimal new camera matrix after rectification
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, image_size, np.eye(3), balance=balance, fov_scale=fov_scale
    )
    
    # Generate rectification maps
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, image_size, cv2.CV_16SC2
    )
    
    return map1, map2, new_K


def rectify_video(input_path, output_path, map1, map2, show_progress=True):
    """
    Rectify entire video using pre-computed maps.
    
    Args:
        input_path: Path to input fisheye video
        output_path: Path to save rectified video
        map1, map2: Rectification maps from cv2.fisheye.initUndistortRectifyMap
        show_progress: Show progress bar
    """
    # Open input video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nInput video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise ValueError(f"Cannot create output video: {output_path}")
    
    # Process frames
    print(f"\nRectifying video...")
    pbar = tqdm(total=total_frames, desc="Processing", disable=not show_progress)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply rectification
        rectified_frame = cv2.remap(
            frame, map1, map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        
        out.write(rectified_frame)
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    print(f"\n✓ Rectification complete!")
    print(f"  Processed {frame_count} frames")
    print(f"  Output saved to: {output_path}")


def create_comparison_samples(input_path, output_dir, map1, map2, num_samples=5):
    """
    Create side-by-side comparison images for visual verification.
    
    Args:
        input_path: Path to input video
        output_dir: Directory to save comparison images
        map1, map2: Rectification maps
        num_samples: Number of sample frames to extract
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(input_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames evenly throughout video
    sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    
    print(f"\nCreating {num_samples} comparison samples...")
    
    for i, frame_idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Rectify frame
        rectified = cv2.remap(
            frame, map1, map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        
        # Create side-by-side comparison
        comparison = np.hstack([frame, rectified])
        
        # Add labels
        cv2.putText(comparison, 'Original (Fisheye)', (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(comparison, 'Rectified (Pinhole)', (frame.shape[1] + 50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # Save
        comparison_path = output_dir / f'comparison_{i:02d}_frame{frame_idx:06d}.jpg'
        cv2.imwrite(str(comparison_path), comparison)
    
    cap.release()
    print(f"  Comparison images saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Rectify fisheye video using saved calibration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python rectify_any_video.py --input raw_video.mp4 --output rectified.mp4
  
  # Specify calibration directory
  python rectify_any_video.py --input video.mp4 --output rect.mp4 --calib-dir ./calibration/
  
  # Create comparison samples without full video
  python rectify_any_video.py --input video.mp4 --samples-only --sample-dir ./samples/
  
  # Adjust rectification parameters
  python rectify_any_video.py --input video.mp4 --output rect.mp4 --balance 0.5
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input fisheye video')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to output rectified video (required unless --samples-only)')
    parser.add_argument('--calib-dir', type=str, 
                       default='./data/caliberation_results',
                       help='Path to calibration directory (default: ./data/caliberation_results)')
    parser.add_argument('--balance', type=float, default=0.0,
                       help='Balance parameter: 0.0=minimize blank areas, 1.0=retain all pixels (default: 0.0)')
    parser.add_argument('--fov-scale', type=float, default=1.0,
                       help='FOV scale factor (default: 1.0)')
    parser.add_argument('--samples-only', action='store_true',
                       help='Only create comparison samples, do not rectify full video')
    parser.add_argument('--sample-dir', type=str, default='./rectification_samples',
                       help='Directory to save comparison samples (default: ./rectification_samples)')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of comparison samples to create (default: 5)')
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress bar')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.samples_only and args.output is None:
        parser.error("--output is required unless --samples-only is specified")
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input video not found: {input_path}")
        sys.exit(1)
    
    calib_dir = Path(args.calib_dir)
    if not calib_dir.exists():
        print(f"ERROR: Calibration directory not found: {calib_dir}")
        sys.exit(1)
    
    print("=" * 70)
    print("FISHEYE VIDEO RECTIFICATION")
    print("=" * 70)
    print(f"Input video: {input_path}")
    if not args.samples_only:
        print(f"Output video: {args.output}")
    print(f"Calibration: {calib_dir}")
    print(f"Balance: {args.balance}")
    print(f"FOV scale: {args.fov_scale}")
    
    # Load calibration
    print("\nLoading calibration...")
    try:
        K, D, image_size = load_calibration(calib_dir)
        print(f"✓ Loaded calibration for {image_size[0]}x{image_size[1]} images")
        print(f"  fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")
    except Exception as e:
        print(f"ERROR loading calibration: {e}")
        sys.exit(1)
    
    # Create rectification maps
    print("\nCreating rectification maps...")
    map1, map2, new_K = create_rectification_maps(
        K, D, image_size, 
        balance=args.balance, 
        fov_scale=args.fov_scale
    )
    print(f"✓ Rectification maps created")
    print(f"  Rectified fx={new_K[0,0]:.2f}, fy={new_K[1,1]:.2f}, cx={new_K[0,2]:.2f}, cy={new_K[1,2]:.2f}")
    
    # Create comparison samples
    create_comparison_samples(
        input_path, 
        args.sample_dir, 
        map1, map2, 
        num_samples=args.num_samples
    )
    
    # Rectify full video if requested
    if not args.samples_only:
        rectify_video(
            input_path, 
            args.output, 
            map1, map2,
            show_progress=not args.no_progress
        )
    else:
        print("\n✓ Sample creation complete (skipping full video as requested)")
    
    print("\n" + "=" * 70)
    print("RECTIFICATION COMPLETE")
    print("=" * 70)
    if not args.samples_only:
        print(f"Rectified video: {args.output}")
    print(f"Comparison samples: {args.sample_dir}")
    print("\nNEXT STEPS:")
    print("1. Check comparison samples to verify rectification quality")
    print("2. Use rectified video for 3D detection inference")
    print("3. Load rectified camera parameters in inference script:")
    print(f"   --intrinsics-json {calib_dir}/rectified_camera_params.json")


if __name__ == "__main__":
    main()