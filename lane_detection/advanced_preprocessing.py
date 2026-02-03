"""
Advanced Preprocessing for Lane Detection
Includes modern techniques for varying conditions (night, rain, fog, etc.)
"""

from typing import Tuple

import cv2
import numpy as np


class AdvancedPreprocessor:
    """Advanced image preprocessing for robust lane detection."""

    def __init__(
        self,
        enable_clahe: bool = True,
        enable_shadow_removal: bool = True,
        enable_night_enhancement: bool = True,
    ):
        """
        Initialize advanced preprocessor.

        Args:
            enable_clahe: Enable CLAHE for better contrast
            enable_shadow_removal: Remove shadows
            enable_night_enhancement: Enhance low-light images
        """
        self.enable_clahe = enable_clahe
        self.enable_shadow_removal = enable_shadow_removal
        self.enable_night_enhancement = enable_night_enhancement

        # Initialize CLAHE
        if self.enable_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply advanced preprocessing pipeline.

        Args:
            image: Input BGR image

        Returns:
            Tuple of (processed_image, mask)
        """
        # Detect scene condition
        is_night = self._is_night_scene(image)

        # Apply appropriate preprocessing
        if is_night and self.enable_night_enhancement:
            processed = self._enhance_night(image)
        else:
            processed = image.copy()

        # Apply CLAHE for better contrast
        if self.enable_clahe:
            processed = self._apply_clahe(processed)

        # Remove shadows if enabled
        if self.enable_shadow_removal:
            processed = self._remove_shadows(processed)

        # Extract lane mask using multi-color-space approach
        mask = self._extract_lane_mask(processed)

        return processed, mask

    def _is_night_scene(self, image: np.ndarray) -> bool:
        """Detect if image is a night scene based on brightness."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        return mean_brightness < 60  # Threshold for night detection

    def _enhance_night(self, image: np.ndarray) -> np.ndarray:
        """Enhance low-light images."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply gamma correction to L channel
        gamma = 1.5
        l_corrected = np.array(255 * (l / 255) ** (1 / gamma), dtype=np.uint8)

        # Merge and convert back
        lab_corrected = cv2.merge([l_corrected, a, b])
        enhanced = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

        return enhanced

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE for adaptive histogram equalization."""
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        l_clahe = self.clahe.apply(l)

        # Merge and convert back
        lab_clahe = cv2.merge([l_clahe, a, b])
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        return result

    def _remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """Remove shadows using morphological operations."""
        # Convert to grayscale for shadow detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Dilate to get shadow regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated = cv2.dilate(gray, kernel)

        # Perform morphological close
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

        # Normalize to get shadow-free image  
        normalized = cv2.divide(gray, closed, scale=255)

        # Apply to all channels
        result = image.copy()
        for i in range(3):
            result[:, :, i] = cv2.normalize(
                cv2.divide(image[:, :, i], closed, scale=255),
                None,
                0,
                255,
                cv2.NORM_MINMAX,
            )

        return result

    def _extract_lane_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Extract lane mask using multi-color-space approach.

        Combines HSV (yellow), LAB (white), and HLS for robust detection.
        """
        # HSV for yellow lanes
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        yellow_lower = np.array([10, 90, 100])
        yellow_upper = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

        # LAB for white lanes
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b_channel = cv2.split(lab)
        white_mask = cv2.threshold(l, 200, 255, cv2.THRESH_BINARY)[1]

        # HLS for additional white detection
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        _, l_hls, s = cv2.split(hls)
        white_hls_mask = cv2.threshold(l_hls, 200, 255, cv2.THRESH_BINARY)[1]

        # Combine all masks
        combined_mask = cv2.bitwise_or(yellow_mask, white_mask)
        combined_mask = cv2.bitwise_or(combined_mask, white_hls_mask)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        return combined_mask


# Convenience function matching existing API
def preprocess_image_advanced(
    image: np.ndarray,
    enable_night_enhancement: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Advanced preprocessing with automatic scene detection.

    Args:
        image: Input BGR image
        enable_night_enhancement: Auto-enhance night scenes

    Returns:
        Tuple of (processed_image, mask)
    """
    preprocessor = AdvancedPreprocessor(
        enable_clahe=True,
        enable_shadow_removal=True,
        enable_night_enhancement=enable_night_enhancement,
    )
    return preprocessor.preprocess(image)
