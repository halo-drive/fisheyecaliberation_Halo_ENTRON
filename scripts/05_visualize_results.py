#!/usr/bin/env python3
"""
Visualize calibration results and quality metrics.

This script creates various visualizations to assess calibration quality
and understand the camera properties.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.export_formats import load_opencv_calibration
from utils.checkerboard_detection import CheckerboardDetector, compute_reprojection_error


def visualize_distortion_pattern(K: np.ndarray, D: np.ndarray, 
                                 image_size: tuple, output_path: Path):
    """
    Visualize fisheye distortion pattern using grid warping.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Create ideal grid
    h, w = image_size[1], image_size[0]
    grid_points = []
    step = 50
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            grid_points.append([x, y])
    
    grid_points = np.array(grid_points, dtype=np.float32).reshape(-1, 1, 2)
    
    # Apply fisheye distortion
    # First, undistort to get normalized coordinates
    normalized = cv2.fisheye.undistortPoints(grid_points, K, D)
    
    # Then project back with zero distortion to see ideal positions
    K_ideal = K.copy()
    D_zero = np.zeros((4, 1))
    distorted = cv2.fisheye.distortPoints(normalized, K_ideal, D_zero)
    
    # Plot original grid
    ax1.set_title('Ideal Undistorted Grid', fontsize=14)
    ax1.scatter(grid_points[:, 0, 0], grid_points[:, 0, 1], c='blue', s=1, alpha=0.5)
    ax1.set_xlim(0, w)
    ax1.set_ylim(h, 0)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    
    # Plot distorted grid with displacement vectors
    ax2.set_title('Fisheye Distortion Visualization', fontsize=14)
    ax2.scatter(distorted[:, 0, 0], distorted[:, 0, 1], c='red', s=1, alpha=0.5)
    
    # Draw displacement vectors (sampled)
    sample_step = 20
    for i in range(0, len(grid_points), sample_step):
        x1, y1 = grid_points[i, 0]
        x2, y2 = distorted[i, 0]
        ax2.arrow(x1, y1, x2-x1, y2-y1, head_width=10, 
                 head_length=15, fc='green', ec='green', alpha=0.3)
    
    ax2.set_xlim(0, w)
    ax2.set_ylim(h, 0)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved distortion visualization: {output_path}")


def visualize_reprojection_errors(video_path: Path,
                                  K: np.ndarray,
                                  D: np.ndarray,
                                  checkerboard_size: tuple,
                                  square_size_mm: float,
                                  output_dir: Path,
                                  max_frames: int = 30):
    """
    Compute and visualize reprojection errors across video.
    """
    detector = CheckerboardDetector(checkerboard_size, square_size_mm)
    
    objpoints = []
    imgpoints = []
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames
    frame_indices = np.linspace(0, total_frames - 1, max_frames * 2, dtype=int)
    
    print("Detecting checkerboards for error analysis...")
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        found, corners = detector.detect(frame)
        if found:
            objpoints.append(detector.get_object_points())
            imgpoints.append(corners)
            
            if len(objpoints) >= max_frames:
                break
    
    cap.release()
    
    if len(objpoints) < 5:
        print("Not enough detections for error analysis")
        return
    
    print(f"Computing reprojection errors for {len(objpoints)} frames...")
    
    # Compute pose for each frame
    rvecs = []
    tvecs = []
    for objp, imgp in zip(objpoints, imgpoints):
        success, rvec, tvec = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            objp.reshape(-1, 1, 3), imgp, K, D, None
        )
        if success:
            rvecs.append(rvec)
            tvecs.append(tvec)
    
    # For visualization, compute errors per point
    all_errors = []
    for i, (objp, imgp) in enumerate(zip(objpoints, imgpoints)):
        if i >= len(rvecs):
            continue
        
        # Project points
        projected, _ = cv2.fisheye.projectPoints(
            objp.reshape(-1, 1, 3), rvecs[i], tvecs[i], K, D
        )
        
        # Compute errors
        errors = np.linalg.norm(imgp - projected, axis=2).flatten()
        all_errors.extend(errors)
    
    all_errors = np.array(all_errors)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram of errors
    axes[0, 0].hist(all_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(all_errors), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(all_errors):.3f}px')
    axes[0, 0].axvline(np.median(all_errors), color='green', linestyle='--',
                       label=f'Median: {np.median(all_errors):.3f}px')
    axes[0, 0].set_xlabel('Reprojection Error (pixels)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Reprojection Errors')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box plot
    axes[0, 1].boxplot(all_errors, vert=True)
    axes[0, 1].set_ylabel('Reprojection Error (pixels)')
    axes[0, 1].set_title('Error Statistics')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error statistics text
    stats_text = f"""
    Error Statistics:
    ─────────────────
    Mean:    {np.mean(all_errors):.4f} px
    Median:  {np.median(all_errors):.4f} px
    Std Dev: {np.std(all_errors):.4f} px
    Min:     {np.min(all_errors):.4f} px
    Max:     {np.max(all_errors):.4f} px
    
    Percentiles:
    25th:    {np.percentile(all_errors, 25):.4f} px
    75th:    {np.percentile(all_errors, 75):.4f} px
    95th:    {np.percentile(all_errors, 95):.4f} px
    
    Total points: {len(all_errors)}
    """
    axes[1, 0].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                   verticalalignment='center')
    axes[1, 0].axis('off')
    
    # Quality assessment
    quality_text = "Calibration Quality Assessment:\n\n"
    mean_err = np.mean(all_errors)
    
    if mean_err < 0.5:
        quality_text += "✓ EXCELLENT (<0.5px)\n"
        quality_text += "  Camera is very well calibrated.\n"
        quality_text += "  Suitable for precise measurements."
    elif mean_err < 1.0:
        quality_text += "✓ GOOD (<1.0px)\n"
        quality_text += "  Camera is well calibrated.\n"
        quality_text += "  Suitable for most applications."
    elif mean_err < 2.0:
        quality_text += "~ ACCEPTABLE (<2.0px)\n"
        quality_text += "  Calibration is reasonable.\n"
        quality_text += "  May need refinement for precision work."
    else:
        quality_text += "✗ POOR (>2.0px)\n"
        quality_text += "  Consider recalibration:\n"
        quality_text += "  - Record more calibration images\n"
        quality_text += "  - Ensure better coverage\n"
        quality_text += "  - Check for camera focus issues"
    
    axes[1, 1].text(0.1, 0.5, quality_text, fontsize=11,
                   verticalalignment='center')
    axes[1, 1].axis('off')
    
    plt.suptitle('Reprojection Error Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / 'reprojection_errors.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved reprojection error analysis: {output_path}")
    print(f"Mean reprojection error: {np.mean(all_errors):.4f} pixels")


def visualize_calibration_summary(calib_dir: Path, output_path: Path):
    """
    Create a comprehensive calibration summary visualization.
    """
    # Load calibration
    K, D, image_size = load_opencv_calibration(calib_dir)
    
    # Load JSON for additional info
    json_path = calib_dir / 'calibration.json'
    additional_info = {}
    if json_path.exists():
        with open(json_path, 'r') as f:
            additional_info = json.load(f)
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Camera matrix visualization
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    
    summary_text = "FISHEYE CAMERA CALIBRATION SUMMARY\n"
    summary_text += "=" * 60 + "\n\n"
    summary_text += f"Image Resolution: {image_size[0]} × {image_size[1]} pixels\n\n"
    summary_text += "Camera Intrinsic Matrix (K):\n"
    summary_text += f"  fx = {K[0,0]:.2f} px    fy = {K[1,1]:.2f} px\n"
    summary_text += f"  cx = {K[0,2]:.2f} px    cy = {K[1,2]:.2f} px\n\n"
    summary_text += "Distortion Coefficients (Fisheye Model):\n"
    summary_text += f"  k1 = {D[0]:.6f}    k2 = {D[1]:.6f}\n"
    summary_text += f"  k3 = {D[2]:.6f}    k4 = {D[3]:.6f}\n\n"
    
    if 'mean_reprojection_error_pixels' in additional_info:
        err = additional_info['mean_reprojection_error_pixels']
        summary_text += f"Mean Reprojection Error: {err:.4f} pixels\n"
    
    ax1.text(0.05, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center')
    
    # FOV visualization
    ax2 = fig.add_subplot(gs[1, 0])
    f_mean = (K[0, 0] + K[1, 1]) / 2
    fov_h = 2 * np.arctan(image_size[0] / (2 * K[0, 0])) * 180 / np.pi
    fov_v = 2 * np.arctan(image_size[1] / (2 * K[1, 1])) * 180 / np.pi
    
    categories = ['Horizontal', 'Vertical']
    fovs = [fov_h, fov_v]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax2.barh(categories, fovs, color=colors, alpha=0.7)
    ax2.set_xlabel('Field of View (degrees)')
    ax2.set_title('Estimated Field of View')
    ax2.grid(True, alpha=0.3, axis='x')
    
    for bar, fov in zip(bars, fovs):
        ax2.text(fov + 2, bar.get_y() + bar.get_height()/2, 
                f'{fov:.1f}°', va='center', fontsize=10, fontweight='bold')
    
    # Principal point visualization
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_xlim(0, image_size[0])
    ax3.set_ylim(image_size[1], 0)
    ax3.set_aspect('equal')
    
    # Draw image bounds
    ax3.add_patch(plt.Rectangle((0, 0), image_size[0], image_size[1],
                                fill=False, edgecolor='black', linewidth=2))
    
    # Draw center
    img_center_x = image_size[0] / 2
    img_center_y = image_size[1] / 2
    ax3.plot(img_center_x, img_center_y, 'b+', markersize=15, markeredgewidth=2,
            label='Image Center')
    
    # Draw principal point
    cx, cy = K[0, 2], K[1, 2]
    ax3.plot(cx, cy, 'rx', markersize=15, markeredgewidth=2,
            label='Principal Point')
    
    # Draw line between them
    ax3.plot([img_center_x, cx], [img_center_y, cy], 'g--', alpha=0.5)
    
    offset_x = cx - img_center_x
    offset_y = cy - img_center_y
    offset_dist = np.sqrt(offset_x**2 + offset_y**2)
    
    ax3.set_title(f'Principal Point Offset: {offset_dist:.1f}px')
    ax3.legend()
    ax3.set_xlabel('X (pixels)')
    ax3.set_ylabel('Y (pixels)')
    ax3.grid(True, alpha=0.3)
    
    # Distortion coefficient plot
    ax4 = fig.add_subplot(gs[2, :])
    coeffs = ['k1', 'k2', 'k3', 'k4']
    values = [D[0], D[1], D[2], D[3]]
    colors = ['#e74c3c' if v > 0 else '#3498db' for v in values]
    
    bars = ax4.bar(coeffs, values, color=colors, alpha=0.7)
    ax4.set_ylabel('Coefficient Value')
    ax4.set_title('Fisheye Distortion Coefficients')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height,
                f'{val:.6f}', ha='center', va='bottom' if val > 0 else 'top',
                fontsize=9)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved calibration summary: {output_path}")


def main():
    # Configuration
    VIDEO_PATH = Path(__file__).parent.parent / "data" / "raw_videos" / "cam0_calib_record2.mp4"
    CALIB_DIR = Path(__file__).parent.parent / "data" / "caliberation_results"
    OUTPUT_DIR = Path(__file__).parent.parent / "data" / "caliberation_results" / "visualizations"
    
    CHECKERBOARD_SIZE = (10, 6)
    SQUARE_SIZE_MM = 75.0
    
    print("=" * 70)
    print("CALIBRATION VISUALIZATION")
    print("=" * 70)
    print()
    
    # Check if calibration exists
    if not (CALIB_DIR / "camera_matrix.npy").exists():
        print("ERROR: Calibration not found!")
        print("Please run 02_caliberation_fisheye.py first")
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load calibration
    print("Loading calibration...")
    K, D, image_size = load_opencv_calibration(CALIB_DIR)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    print("1. Creating calibration summary...")
    visualize_calibration_summary(CALIB_DIR, OUTPUT_DIR / 'summary.png')
    
    print("2. Creating distortion pattern visualization...")
    visualize_distortion_pattern(K, D, image_size, OUTPUT_DIR / 'distortion_pattern.png')
    
    if VIDEO_PATH.exists():
        print("3. Analyzing reprojection errors...")
        visualize_reprojection_errors(VIDEO_PATH, K, D, CHECKERBOARD_SIZE, 
                                     SQUARE_SIZE_MM, OUTPUT_DIR, max_frames=30)
    
    print()
    print("=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"Visualizations saved to: {OUTPUT_DIR}/")
    print()
    print("Generated files:")
    print("  - summary.png: Overall calibration summary")
    print("  - distortion_pattern.png: Fisheye distortion visualization")
    print("  - reprojection_errors.png: Error analysis")
    print()


if __name__ == "__main__":
    main()