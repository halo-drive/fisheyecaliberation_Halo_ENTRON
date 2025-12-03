"""
Export calibration results to various formats.
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any


def save_opencv_format(K: np.ndarray, D: np.ndarray, 
                      image_size: tuple, output_path: Path) -> None:
    """
    Save calibration in OpenCV numpy format.
    
    Args:
        K: Camera intrinsic matrix (3x3)
        D: Distortion coefficients (4x1 for fisheye)
        image_size: (width, height)
        output_path: Path to save directory
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path / 'camera_matrix.npy', K)
    np.save(output_path / 'distortion_coeffs.npy', D)
    np.save(output_path / 'image_size.npy', np.array(image_size))
    
    print(f"Saved OpenCV format to {output_path}/")


def save_json_format(K: np.ndarray, D: np.ndarray,
                    image_size: tuple, output_path: Path,
                    additional_info: Dict[str, Any] = None) -> None:
    """
    Save calibration in JSON format.
    
    Args:
        K: Camera intrinsic matrix (3x3)
        D: Distortion coefficients
        image_size: (width, height)
        output_path: Path to JSON file
        additional_info: Additional metadata to include
    """
    calibration_data = {
        "camera_matrix": {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "data": K.tolist()
        },
        "distortion_coefficients": {
            "k1": float(D[0]),
            "k2": float(D[1]),
            "k3": float(D[2]),
            "k4": float(D[3]),
            "data": D.flatten().tolist()
        },
        "image_size": {
            "width": int(image_size[0]),
            "height": int(image_size[1])
        },
        "distortion_model": "fisheye"
    }
    
    if additional_info:
        calibration_data.update(additional_info)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(calibration_data, f, indent=2)
    
    print(f"Saved JSON format to {output_path}")


def save_driveworks_compatible(K: np.ndarray, D: np.ndarray,
                               image_size: tuple, output_path: Path) -> None:
    """
    Save calibration in DriveWorks-compatible format (ftheta model approximation).
    
    Note: This is an approximation. OpenCV fisheye uses equidistant projection,
    while DriveWorks ftheta uses f*theta model. For small angles they're similar.
    
    Args:
        K: Camera intrinsic matrix (3x3)
        D: Distortion coefficients (4x1)
        image_size: (width, height)
        output_path: Path to JSON file
    """
    # DriveWorks ftheta format
    dw_data = {
        "model": "ftheta",
        "image-size": [int(image_size[0]), int(image_size[1])],
        "focalLengthPixel": [float(K[0, 0]), float(K[1, 1])],
        "principalPoint": [float(K[0, 2]), float(K[1, 2])],
        "distortion": {
            "k1": float(D[0]),
            "k2": float(D[1]),
            "k3": float(D[2]),
            "k4": float(D[3])
        },
        "note": "Converted from OpenCV fisheye calibration. May need refinement in DriveWorks."
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(dw_data, f, indent=2)
    
    print(f"Saved DriveWorks-compatible format to {output_path}")


def save_calibration_report(K: np.ndarray, D: np.ndarray,
                           image_size: tuple,
                           mean_error: float,
                           per_view_errors: list,
                           num_images: int,
                           output_path: Path) -> None:
    """
    Save a human-readable calibration report.
    
    Args:
        K: Camera intrinsic matrix
        D: Distortion coefficients
        image_size: (width, height)
        mean_error: Mean reprojection error
        per_view_errors: List of per-view errors
        num_images: Number of calibration images used
        output_path: Path to text file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = []
    report.append("=" * 70)
    report.append("FISHEYE CAMERA CALIBRATION REPORT")
    report.append("=" * 70)
    report.append("")
    
    report.append("IMAGE INFORMATION:")
    report.append(f"  Resolution: {image_size[0]} x {image_size[1]} pixels")
    report.append(f"  Number of calibration images: {num_images}")
    report.append("")
    
    report.append("CAMERA INTRINSIC MATRIX (K):")
    report.append(f"  fx = {K[0, 0]:.4f} pixels")
    report.append(f"  fy = {K[1, 1]:.4f} pixels")
    report.append(f"  cx = {K[0, 2]:.4f} pixels")
    report.append(f"  cy = {K[1, 2]:.4f} pixels")
    report.append("")
    report.append("  Full matrix:")
    for row in K:
        report.append(f"    [{row[0]:12.6f} {row[1]:12.6f} {row[2]:12.6f}]")
    report.append("")
    
    report.append("DISTORTION COEFFICIENTS (fisheye model):")
    report.append(f"  k1 = {D[0]:.8f}")
    report.append(f"  k2 = {D[1]:.8f}")
    report.append(f"  k3 = {D[2]:.8f}")
    report.append(f"  k4 = {D[3]:.8f}")
    report.append("")
    
    report.append("CALIBRATION QUALITY:")
    report.append(f"  Mean reprojection error: {mean_error:.4f} pixels")
    report.append(f"  Min error: {min(per_view_errors):.4f} pixels")
    report.append(f"  Max error: {max(per_view_errors):.4f} pixels")
    report.append(f"  Std deviation: {np.std(per_view_errors):.4f} pixels")
    report.append("")
    
    quality = "EXCELLENT" if mean_error < 0.5 else \
              "GOOD" if mean_error < 1.0 else \
              "ACCEPTABLE" if mean_error < 2.0 else "POOR"
    report.append(f"  Overall quality: {quality}")
    report.append("")
    
    report.append("FIELD OF VIEW ESTIMATES:")
    # Rough FOV estimation for fisheye
    diagonal_pixels = np.sqrt(image_size[0]**2 + image_size[1]**2)
    f_mean = (K[0, 0] + K[1, 1]) / 2
    fov_diagonal = 2 * np.arctan(diagonal_pixels / (2 * f_mean)) * 180 / np.pi
    fov_horizontal = 2 * np.arctan(image_size[0] / (2 * K[0, 0])) * 180 / np.pi
    fov_vertical = 2 * np.arctan(image_size[1] / (2 * K[1, 1])) * 180 / np.pi
    
    report.append(f"  Horizontal FOV: ~{fov_horizontal:.1f}°")
    report.append(f"  Vertical FOV: ~{fov_vertical:.1f}°")
    report.append(f"  Diagonal FOV: ~{fov_diagonal:.1f}°")
    report.append("")
    
    report.append("=" * 70)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Saved calibration report to {output_path}")


def load_opencv_calibration(input_path: Path) -> tuple:
    """
    Load calibration from OpenCV numpy format.
    
    Args:
        input_path: Path to directory containing .npy files
        
    Returns:
        (K, D, image_size) tuple
    """
    input_path = Path(input_path)
    
    K = np.load(input_path / 'camera_matrix.npy')
    D = np.load(input_path / 'distortion_coeffs.npy')
    image_size = tuple(np.load(input_path / 'image_size.npy'))
    
    return K, D, image_size