#!/usr/bin/env python3
"""
Quick Start Script for Lane Detection System
Minimal dependencies demo that works immediately.
"""

import cv2
import numpy as np
from pathlib import Path

def quick_lane_detection(image_path: str, output_path: str = None):
    """
    Quick lane detection using traditional CV (no deep learning required).
    
    Args:
        image_path: Path to input image
        output_path: Optional output path (default: adds '_detected' suffix)
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not read image: {image_path}")
        return
    
    print(f"✓ Loaded image: {image.shape}")
    
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for yellow and white lanes
    # Yellow
    yellow_lower = np.array([10, 90, 100])
    yellow_upper = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    
    # White
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    
    # Combine masks
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    print("✓ Created lane mask")
    
    # Detect edges
    edges = cv2.Canny(mask, 50, 150)
    
    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=20,
        minLineLength=20,
        maxLineGap=300
    )
    
    # Draw lines on image
    result = image.copy()
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
        print(f"✓ Detected {len(lines)} lane segments")
    else:
        print("⚠ No lanes detected")
    
    # Determine output path
    if output_path is None:
        path = Path(image_path)
        output_path = str(path.parent / f"{path.stem}_detected{path.suffix}")
    
    # Save result
    cv2.imwrite(output_path, result)
    print(f"✓ Saved result to: {output_path}")
    
    return result, mask


if __name__ == "__main__":
    import sys
    
    print("="*50)
    print("Quick Lane Detection Demo")
    print("="*50)
    print()
    
    if len(sys.argv) < 2:
        print("Usage: python quick_start.py <image_path>")
        print()
        print("Example:")
        print("  python quick_start.py data/test_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    quick_lane_detection(image_path, output_path)
    
    print()
    print("="*50)
    print("Done! Check the output image.")
    print("="*50)
