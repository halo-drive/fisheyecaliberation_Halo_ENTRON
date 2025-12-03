"""
Checkerboard detection utilities for fisheye camera calibration.
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List


class CheckerboardDetector:
    """Handles checkerboard detection in images."""
    
    def __init__(self, checkerboard_size: Tuple[int, int], square_size_mm: float):
        """
        Initialize checkerboard detector.
        
        Args:
            checkerboard_size: (cols, rows) - number of internal corners
            square_size_mm: Size of each square in millimeters
        """
        self.checkerboard_size = checkerboard_size
        self.square_size_mm = square_size_mm
        
        # Prepare 3D object points (in mm)
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 
                                     0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size_mm
        
        # Termination criteria for corner refinement
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    def detect(self, frame: np.ndarray, 
               refine_corners: bool = True,
               blur_threshold: Optional[float] = None) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Detect checkerboard in frame.
        
        Args:
            frame: Input image (color or grayscale)
            refine_corners: Whether to refine corner positions using cornerSubPix
            blur_threshold: If provided, reject frames with Laplacian variance below threshold
            
        Returns:
            (success, corners) tuple where corners are refined 2D positions if found
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Check for blur if threshold provided
        if blur_threshold is not None:
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < blur_threshold:
                return False, None
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
        
        if ret and refine_corners:
            # Refine corner positions to subpixel accuracy
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
        
        return ret, corners
    
    def draw_corners(self, frame: np.ndarray, corners: np.ndarray, 
                     found: bool = True) -> np.ndarray:
        """
        Draw detected corners on frame.
        
        Args:
            frame: Input image
            corners: Detected corner positions
            found: Whether detection was successful
            
        Returns:
            Frame with drawn corners
        """
        output = frame.copy()
        cv2.drawChessboardCorners(output, self.checkerboard_size, corners, found)
        return output
    
    def get_object_points(self) -> np.ndarray:
        """Get the 3D object points for this checkerboard."""
        return self.objp.copy()


def compute_reprojection_error(objpoints: List[np.ndarray],
                               imgpoints: List[np.ndarray],
                               rvecs: List[np.ndarray],
                               tvecs: List[np.ndarray],
                               K: np.ndarray,
                               D: np.ndarray) -> Tuple[float, List[float]]:
    """
    Compute reprojection error for fisheye calibration.
    
    Args:
        objpoints: List of 3D object points
        imgpoints: List of 2D image points
        rvecs: Rotation vectors
        tvecs: Translation vectors
        K: Camera intrinsic matrix
        D: Distortion coefficients
        
    Returns:
        (mean_error, per_view_errors) tuple
    """
    errors = []
    
    for i in range(len(objpoints)):
        # Project 3D points to 2D
        imgpoints2, _ = cv2.fisheye.projectPoints(
            objpoints[i].reshape(-1, 1, 3),
            rvecs[i],
            tvecs[i],
            K,
            D
        )
        
        # Compute error
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        errors.append(error)
    
    mean_error = np.mean(errors)
    return mean_error, errors


def is_good_checkerboard_pose(corners: np.ndarray, 
                              image_shape: Tuple[int, int],
                              min_area_ratio: float = 0.05,
                              max_area_ratio: float = 0.8) -> bool:
    """
    Check if detected checkerboard has good pose for calibration.
    
    Args:
        corners: Detected corner positions
        image_shape: (height, width) of image
        min_area_ratio: Minimum ratio of checkerboard area to image area
        max_area_ratio: Maximum ratio of checkerboard area to image area
        
    Returns:
        True if pose is good for calibration
    """
    # Compute bounding box area
    x_coords = corners[:, 0, 0]
    y_coords = corners[:, 0, 1]
    
    bbox_area = (x_coords.max() - x_coords.min()) * (y_coords.max() - y_coords.min())
    image_area = image_shape[0] * image_shape[1]
    
    area_ratio = bbox_area / image_area
    
    return min_area_ratio <= area_ratio <= max_area_ratio