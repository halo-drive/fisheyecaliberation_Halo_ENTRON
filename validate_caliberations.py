#!/usr/bin/env python3
"""
Compare OpenCV and DriveWorks Calibrations

This script compares the calibration results from:
  1. OpenCV fisheye calibration (your pipeline)
  2. DriveWorks calibration (cam-2.json)

It analyzes differences in intrinsics, distortion, and quality metrics.
"""
import json
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt


def load_opencv_calibration(calib_dir):
    """Load OpenCV calibration results."""
    calib_dir = Path(calib_dir)
    
    # Load from JSON (has all info)
    with open(calib_dir / 'calibration.json', 'r') as f:
        opencv_data = json.load(f)
    
    # Also load rectified params
    with open(calib_dir / 'rectified_camera_params.json', 'r') as f:
        rectified_data = json.load(f)
    
    # Extract camera matrix
    K_original = np.array(opencv_data['camera_matrix']['data'])
    
    # Extract distortion (OpenCV fisheye: 4 coefficients)
    D_original = np.array(opencv_data['distortion_coefficients']['data']).reshape(-1, 1)
    
    # Rectified intrinsics
    K_rectified = np.array([
        [rectified_data['rectified_intrinsics']['fx'], 0, rectified_data['rectified_intrinsics']['cx']],
        [0, rectified_data['rectified_intrinsics']['fy'], rectified_data['rectified_intrinsics']['cy']],
        [0, 0, 1]
    ])
    
    image_size = (opencv_data['image_size']['width'], opencv_data['image_size']['height'])
    
    return {
        'name': 'OpenCV Fisheye',
        'model': 'equidistant (Kannala-Brandt)',
        'K_original': K_original,
        'K_rectified': K_rectified,
        'D': D_original,
        'image_size': image_size,
        'num_images': opencv_data.get('num_images', 'unknown'),
        'reprojection_error': opencv_data.get('mean_reprojection_error_pixels', None),
        'raw_data': opencv_data,
        'rectified_data': rectified_data
    }


def load_driveworks_calibration(dw_json_path):
    """Load DriveWorks calibration from constraints JSON."""
    with open(dw_json_path, 'r') as f:
        dw_data = json.load(f)
    
    # DW stores constraints, not final calibration
    # We need to check if it has the calibrated model
    
    num_constraints = len(dw_data.get('constraints', []))
    image_size = tuple(dw_data.get('image-size', [None, None]))
    
    # Check if model parameters are present
    K_dw = None
    D_dw = None
    model_type = dw_data.get('type', 'unknown')
    
    # Try to extract if calibration was run
    if 'focalLengthPixel' in dw_data and dw_data['focalLengthPixel'] is not None:
        fx, fy = dw_data['focalLengthPixel']
        K_dw = np.array([
            [fx, 0, image_size[0]/2],
            [0, fy, image_size[1]/2],
            [0, 0, 1]
        ])
    
    # Note: DW JSON might not have final calibration if it's just constraints
    
    return {
        'name': 'DriveWorks',
        'model': model_type,
        'K_original': K_dw,
        'D': D_dw,
        'image_size': image_size,
        'num_constraints': num_constraints,
        'has_calibration': K_dw is not None,
        'raw_data': dw_data
    }


def compare_intrinsics(opencv_calib, dw_calib):
    """Compare camera intrinsics."""
    print("\n" + "="*70)
    print("INTRINSICS COMPARISON")
    print("="*70)
    
    # OpenCV
    K_cv = opencv_calib['K_original']
    print(f"\n{opencv_calib['name']} - Original Fisheye K:")
    print(f"  fx = {K_cv[0, 0]:.2f} pixels")
    print(f"  fy = {K_cv[1, 1]:.2f} pixels")
    print(f"  cx = {K_cv[0, 2]:.2f} pixels")
    print(f"  cy = {K_cv[1, 2]:.2f} pixels")
    print(f"  Model: {opencv_calib['model']}")
    
    # OpenCV Rectified
    K_rect = opencv_calib['K_rectified']
    print(f"\n{opencv_calib['name']} - Rectified K:")
    print(f"  fx = {K_rect[0, 0]:.2f} pixels")
    print(f"  fy = {K_rect[1, 1]:.2f} pixels")
    print(f"  cx = {K_rect[0, 2]:.2f} pixels")
    print(f"  cy = {K_rect[1, 2]:.2f} pixels")
    
    # DriveWorks
    if dw_calib['has_calibration']:
        K_dw = dw_calib['K_original']
        print(f"\n{dw_calib['name']} - K:")
        print(f"  fx = {K_dw[0, 0]:.2f} pixels")
        print(f"  fy = {K_dw[1, 1]:.2f} pixels")
        print(f"  cx = {K_dw[0, 2]:.2f} pixels")
        print(f"  cy = {K_dw[1, 2]:.2f} pixels")
        print(f"  Model: {dw_calib['model']}")
        
        # Compute differences
        print(f"\nDifferences (OpenCV - DriveWorks):")
        print(f"  Δfx = {K_cv[0, 0] - K_dw[0, 0]:.2f} pixels ({abs(K_cv[0, 0] - K_dw[0, 0])/K_dw[0, 0]*100:.1f}%)")
        print(f"  Δfy = {K_cv[1, 1] - K_dw[1, 1]:.2f} pixels ({abs(K_cv[1, 1] - K_dw[1, 1])/K_dw[1, 1]*100:.1f}%)")
        print(f"  Δcx = {K_cv[0, 2] - K_dw[0, 2]:.2f} pixels ({abs(K_cv[0, 2] - K_dw[0, 2])/K_dw[0, 2]*100:.1f}%)")
        print(f"  Δcy = {K_cv[1, 2] - K_dw[1, 2]:.2f} pixels ({abs(K_cv[1, 2] - K_dw[1, 2])/K_dw[1, 2]*100:.1f}%)")
    else:
        print(f"\n{dw_calib['name']}: Calibration parameters not found in JSON")
        print("  (This file might only contain constraints, not final calibration)")


def compare_distortion(opencv_calib, dw_calib):
    """Compare distortion coefficients."""
    print("\n" + "="*70)
    print("DISTORTION COMPARISON")
    print("="*70)
    
    # OpenCV
    D_cv = opencv_calib['D'].flatten()
    print(f"\n{opencv_calib['name']} (Equidistant model):")
    print(f"  k1 = {D_cv[0]:.8f}")
    print(f"  k2 = {D_cv[1]:.8f}")
    print(f"  k3 = {D_cv[2]:.8f}")
    print(f"  k4 = {D_cv[3]:.8f}")
    
    # DriveWorks
    if dw_calib['D'] is not None:
        D_dw = dw_calib['D']
        print(f"\n{dw_calib['name']} ({dw_calib['model']} model):")
        for i, coeff in enumerate(D_dw):
            print(f"  k{i+1} = {coeff:.8f}")
    else:
        print(f"\n{dw_calib['name']}: Distortion coefficients not available")
    
    print("\nNOTE: Different distortion models (equidistant vs ftheta)")
    print("      Direct coefficient comparison not meaningful!")
    print("      Better to compare rectification quality visually.")


def compare_metadata(opencv_calib, dw_calib):
    """Compare calibration metadata."""
    print("\n" + "="*70)
    print("CALIBRATION METADATA")
    print("="*70)
    
    print(f"\n{opencv_calib['name']}:")
    print(f"  Images/frames used: {opencv_calib['num_images']}")
    print(f"  Image size: {opencv_calib['image_size'][0]}x{opencv_calib['image_size'][1]}")
    if opencv_calib['reprojection_error']:
        print(f"  Reprojection error: {opencv_calib['reprojection_error']:.4f} pixels")
    print(f"  Distortion model: {opencv_calib['model']}")
    
    print(f"\n{dw_calib['name']}:")
    print(f"  Constraints used: {dw_calib['num_constraints']}")
    print(f"  Image size: {dw_calib['image_size'][0]}x{dw_calib['image_size'][1]}")
    print(f"  Distortion model: {dw_calib['model']}")
    print(f"  Has final calibration: {'Yes' if dw_calib['has_calibration'] else 'No (constraints only)'}")


def compute_fov_estimates(K, image_size):
    """Estimate field of view from intrinsics."""
    fx, fy = K[0, 0], K[1, 1]
    w, h = image_size
    
    fov_h = 2 * np.arctan(w / (2 * fx)) * 180 / np.pi
    fov_v = 2 * np.arctan(h / (2 * fy)) * 180 / np.pi
    fov_d = 2 * np.arctan(np.sqrt(w**2 + h**2) / (2 * ((fx + fy) / 2))) * 180 / np.pi
    
    return fov_h, fov_v, fov_d


def compare_fov(opencv_calib, dw_calib):
    """Compare field of view estimates."""
    print("\n" + "="*70)
    print("FIELD OF VIEW ESTIMATES")
    print("="*70)
    
    # OpenCV Original
    fov_h, fov_v, fov_d = compute_fov_estimates(
        opencv_calib['K_original'], 
        opencv_calib['image_size']
    )
    print(f"\n{opencv_calib['name']} - Original:")
    print(f"  Horizontal FOV: {fov_h:.1f}°")
    print(f"  Vertical FOV:   {fov_v:.1f}°")
    print(f"  Diagonal FOV:   {fov_d:.1f}°")
    
    # OpenCV Rectified
    fov_h_rect, fov_v_rect, fov_d_rect = compute_fov_estimates(
        opencv_calib['K_rectified'],
        opencv_calib['image_size']
    )
    print(f"\n{opencv_calib['name']} - Rectified:")
    print(f"  Horizontal FOV: {fov_h_rect:.1f}°")
    print(f"  Vertical FOV:   {fov_v_rect:.1f}°")
    print(f"  Diagonal FOV:   {fov_d_rect:.1f}°")
    
    # DriveWorks
    if dw_calib['has_calibration']:
        fov_h_dw, fov_v_dw, fov_d_dw = compute_fov_estimates(
            dw_calib['K_original'],
            dw_calib['image_size']
        )
        print(f"\n{dw_calib['name']}:")
        print(f"  Horizontal FOV: {fov_h_dw:.1f}°")
        print(f"  Vertical FOV:   {fov_v_dw:.1f}°")
        print(f"  Diagonal FOV:   {fov_d_dw:.1f}°")
        
        print(f"\nFOV Differences (OpenCV - DriveWorks):")
        print(f"  ΔHorizontal: {fov_h - fov_h_dw:.1f}°")
        print(f"  ΔVertical:   {fov_v - fov_v_dw:.1f}°")


def visualize_comparison(opencv_calib, dw_calib, output_path):
    """Create visualization comparing both calibrations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Intrinsics comparison (bar chart)
    ax = axes[0, 0]
    params = ['fx', 'fy', 'cx', 'cy']
    
    K_cv = opencv_calib['K_original']
    cv_values = [K_cv[0, 0], K_cv[1, 1], K_cv[0, 2], K_cv[1, 2]]
    
    if dw_calib['has_calibration']:
        K_dw = dw_calib['K_original']
        dw_values = [K_dw[0, 0], K_dw[1, 1], K_dw[0, 2], K_dw[1, 2]]
        
        x = np.arange(len(params))
        width = 0.35
        
        ax.bar(x - width/2, cv_values, width, label='OpenCV', alpha=0.8)
        ax.bar(x + width/2, dw_values, width, label='DriveWorks', alpha=0.8)
        ax.set_ylabel('Pixels')
        ax.set_title('Intrinsics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(params)
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.bar(params, cv_values, alpha=0.8)
        ax.set_ylabel('Pixels')
        ax.set_title('OpenCV Intrinsics (DW Not Available)')
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Distortion coefficients
    ax = axes[0, 1]
    D_cv = opencv_calib['D'].flatten()
    coeff_names = ['k1', 'k2', 'k3', 'k4']
    
    ax.bar(coeff_names, D_cv, alpha=0.8, color='green')
    ax.set_ylabel('Coefficient Value')
    ax.set_title('OpenCV Distortion Coefficients')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: FOV comparison
    ax = axes[1, 0]
    fov_h, fov_v, fov_d = compute_fov_estimates(opencv_calib['K_original'], opencv_calib['image_size'])
    
    categories = ['Horizontal', 'Vertical', 'Diagonal']
    cv_fovs = [fov_h, fov_v, fov_d]
    
    if dw_calib['has_calibration']:
        fov_h_dw, fov_v_dw, fov_d_dw = compute_fov_estimates(dw_calib['K_original'], dw_calib['image_size'])
        dw_fovs = [fov_h_dw, fov_v_dw, fov_d_dw]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, cv_fovs, width, label='OpenCV', alpha=0.8)
        ax.bar(x + width/2, dw_fovs, width, label='DriveWorks', alpha=0.8)
    else:
        ax.bar(categories, cv_fovs, alpha=0.8)
    
    ax.set_ylabel('Degrees')
    ax.set_title('Field of View Estimates')
    ax.set_xticks(x if dw_calib['has_calibration'] else range(len(categories)))
    ax.set_xticklabels(categories)
    if dw_calib['has_calibration']:
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary text
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = "CALIBRATION SUMMARY\n\n"
    summary_text += f"OpenCV:\n"
    summary_text += f"  Detections: {opencv_calib['num_images']}\n"
    if opencv_calib['reprojection_error']:
        summary_text += f"  Error: {opencv_calib['reprojection_error']:.4f} px\n"
    summary_text += f"  Model: {opencv_calib['model']}\n\n"
    
    summary_text += f"DriveWorks:\n"
    summary_text += f"  Constraints: {dw_calib['num_constraints']}\n"
    summary_text += f"  Model: {dw_calib['model']}\n"
    summary_text += f"  Calibrated: {'Yes' if dw_calib['has_calibration'] else 'No'}\n\n"
    
    if dw_calib['has_calibration']:
        summary_text += "Differences:\n"
        K_cv = opencv_calib['K_original']
        K_dw = dw_calib['K_original']
        summary_text += f"  Δfx: {abs(K_cv[0,0] - K_dw[0,0]):.1f} px\n"
        summary_text += f"  Δfy: {abs(K_cv[1,1] - K_dw[1,1]):.1f} px\n"
        summary_text += f"  Δcx: {abs(K_cv[0,2] - K_dw[0,2]):.1f} px\n"
        summary_text += f"  Δcy: {abs(K_cv[1,2] - K_dw[1,2]):.1f} px\n"
    
    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
           verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()


def main():
    # Paths
    OPENCV_CALIB_DIR = Path("./data/caliberation_results")
    DW_JSON_PATH = Path("./data/caliberation_results/driveworks_Constraints.json")
    OUTPUT_DIR = Path("./calibration_validation")
    
    print("="*70)
    print("CALIBRATION VALIDATION: OpenCV vs DriveWorks")
    print("="*70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load calibrations
    print("\nLoading calibrations...")
    
    try:
        opencv_calib = load_opencv_calibration(OPENCV_CALIB_DIR)
        print(f"✓ Loaded OpenCV calibration")
    except Exception as e:
        print(f"✗ Error loading OpenCV calibration: {e}")
        return 1
    
    try:
        dw_calib = load_driveworks_calibration(DW_JSON_PATH)
        print(f"✓ Loaded DriveWorks constraints")
        if not dw_calib['has_calibration']:
            print("  ⚠ Warning: DW JSON contains only constraints, not final calibration")
    except Exception as e:
        print(f"✗ Error loading DriveWorks calibration: {e}")
        return 1
    
    # Run comparisons
    compare_metadata(opencv_calib, dw_calib)
    compare_intrinsics(opencv_calib, dw_calib)
    compare_distortion(opencv_calib, dw_calib)
    compare_fov(opencv_calib, dw_calib)
    
    # Create visualization
    visualize_comparison(opencv_calib, dw_calib, OUTPUT_DIR / 'calibration_comparison.png')
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if not dw_calib['has_calibration']:
        print("\n⚠ DriveWorks calibration parameters not found!")
        print("  The cam-2.json file contains constraints but not the final")
        print("  calibrated camera model.")
        print("\n  To complete comparison, you need to:")
        print("  1. Run DriveWorks calibration tool to completion")
        print("  2. Export the final calibrated camera parameters")
        print("  3. Re-run this comparison")
        print("\n  OR: Use the next script to calibrate with OpenCV using")
        print("      the same 32 constraints DW used (apples-to-apples)")
    else:
        K_cv = opencv_calib['K_original']
        K_dw = dw_calib['K_original']
        
        fx_diff_pct = abs(K_cv[0,0] - K_dw[0,0]) / K_dw[0,0] * 100
        
        print(f"\nIntrinsics difference: {fx_diff_pct:.1f}%")
        
        if fx_diff_pct < 2:
            print("✓ Calibrations are VERY SIMILAR")
            print("  Either calibration should work well")
        elif fx_diff_pct < 5:
            print("~ Calibrations are REASONABLY SIMILAR")
            print("  Test both to see which gives better 3D detection")
        else:
            print("✗ Calibrations are SIGNIFICANTLY DIFFERENT")
            print("  Investigate which is more accurate")
            print("  Consider visual rectification quality test")
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    print(f"Results saved to: {OUTPUT_DIR}/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
