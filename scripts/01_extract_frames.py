#!/usr/bin/env python3
"""
Extract frames from calibration video with checkerboard detection.

This script processes the calibration video and extracts frames where the
checkerboard is successfully detected. It helps reduce the dataset to only
useful frames for calibration.
"""
import cv2
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.checkerboard_detection import CheckerboardDetector, is_good_checkerboard_pose


def main():
    # Configuration
    VIDEO_PATH = Path(__file__).parent.parent / "data" / "raw_videos" / "cam0_calib_record2.mp4"
    OUTPUT_DIR = Path(__file__).parent.parent / "data" / "extracted_frames"
    
    # Checkerboard parameters
    CHECKERBOARD_SIZE = (10, 6)  # Internal corners (cols, rows)
    SQUARE_SIZE_MM = 75.0
    
    # Detection parameters
    BLUR_THRESHOLD = 100.0  # Reject blurry frames
    MIN_AREA_RATIO = 0.05   # Minimum checkerboard area
    MAX_AREA_RATIO = 0.8    # Maximum checkerboard area
    SKIP_FRAMES = 10        # Process every Nth frame to speed up
    
    print("=" * 70)
    print("FRAME EXTRACTION FROM CALIBRATION VIDEO")
    print("=" * 70)
    print(f"Video: {VIDEO_PATH}")
    print(f"Checkerboard: {CHECKERBOARD_SIZE[0]}x{CHECKERBOARD_SIZE[1]} internal corners")
    print(f"Square size: {SQUARE_SIZE_MM} mm")
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    detector = CheckerboardDetector(CHECKERBOARD_SIZE, SQUARE_SIZE_MM)
    
    # Open video
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {VIDEO_PATH}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Resolution: {width}x{height}")
    print()
    
    # Process video
    frame_count = 0
    extracted_count = 0
    rejected_blur = 0
    rejected_pose = 0
    rejected_no_detection = 0
    
    print("Processing video...")
    pbar = tqdm(total=total_frames // SKIP_FRAMES, desc="Extracting frames")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames to speed up processing
        if frame_count % SKIP_FRAMES != 0:
            frame_count += 1
            continue
        
        # Detect checkerboard
        found, corners = detector.detect(frame, 
                                        refine_corners=True,
                                        blur_threshold=BLUR_THRESHOLD)
        
        if not found:
            rejected_no_detection += 1
        elif not is_good_checkerboard_pose(corners, (height, width), 
                                          MIN_AREA_RATIO, MAX_AREA_RATIO):
            rejected_pose += 1
        else:
            # Save frame and visualization
            frame_filename = f"frame_{frame_count:06d}.jpg"
            frame_viz_filename = f"frame_{frame_count:06d}_viz.jpg"
            
            # Save original frame
            cv2.imwrite(str(OUTPUT_DIR / frame_filename), frame)
            
            # Save visualization with detected corners
            viz_frame = detector.draw_corners(frame, corners, found=True)
            cv2.imwrite(str(OUTPUT_DIR / frame_viz_filename), viz_frame)
            
            extracted_count += 1
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    print()
    print("=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Frames processed: {frame_count}")
    print(f"Frames extracted: {extracted_count}")
    print(f"Rejected (no detection): {rejected_no_detection}")
    print(f"Rejected (poor pose): {rejected_pose}")
    print()
    print(f"Extracted frames saved to: {OUTPUT_DIR}")
    print()
    
    if extracted_count < 20:
        print("WARNING: Less than 20 frames extracted. This may not be enough")
        print("         for robust fisheye calibration. Consider:")
        print("         - Lowering BLUR_THRESHOLD")
        print("         - Adjusting MIN_AREA_RATIO / MAX_AREA_RATIO")
        print("         - Recording a new video with better checkerboard coverage")


if __name__ == "__main__":
    main()