"""
Lane Detection System - Modern Implementation
Version 2.0.0 - 2026 Edition
"""

__version__ = "2.0.0"
__author__ = "Navnit"

from .advanced_preprocessing import AdvancedPreprocessor, preprocess_image_advanced
from .detector import LaneDetector
from .models import YOLOLaneDetector, detect_lanes_yolo
from .preprocessor import preprocess_image
from .tracker import LaneTracker
from .utils import detect_hough_lines, draw_lane_lines

__all__ = [
    "LaneDetector",
    "YOLOLaneDetector",
    "detect_lanes_yolo",
    "preprocess_image",
    "preprocess_image_advanced",
    "AdvancedPreprocessor",
    "LaneTracker",
    "detect_hough_lines",
    "draw_lane_lines",
]
