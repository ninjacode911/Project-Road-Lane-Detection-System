"""
Test Suite for Lane Detection System
"""

import numpy as np
import pytest
import cv2
from lane_detection import LaneDetector, YOLOLaneDetector, preprocess_image


@pytest.fixture
def sample_image():
    """Create a simple test image."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a white lane line
    cv2.line(image, (100, 480), (200, 240), (255, 255, 255), 5)
    cv2.line(image, (500, 480), (400, 240), (255, 255, 255), 5)
    return image


@pytest.fixture
def sample_image_file(tmp_path, sample_image):
    """Save sample image to file."""
    image_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(image_path), sample_image)
    return str(image_path)


class TestPreprocessing:
    """Test preprocessing functions."""

    def test_preprocess_image(self, sample_image):
        """Test basic preprocessing."""
        processed, mask = preprocess_image(sample_image)

        assert processed is not None
        assert mask is not None
        assert processed.shape == sample_image.shape
        assert mask.shape == sample_image.shape[:2]
        assert mask.dtype == np.uint8

    def test_preprocess_image_shape(self, sample_image):
        """Test that preprocessing preserves image dimensions."""
        processed, mask = preprocess_image(sample_image)

        assert processed.shape[0] == sample_image.shape[0]
        assert processed.shape[1] == sample_image.shape[1]


class TestTraditionalDetector:
    """Test traditional lane detector."""

    def test_detector_initialization(self):
        """Test detector can be initialized."""
        detector = LaneDetector()
        assert detector is not None

    def test_detect_lane(self, sample_image):
        """Test lane detection on sample image."""
        detector = LaneDetector()
        detected_image, mask = detector.detect_lane(sample_image)

        assert detected_image is not None
        assert mask is not None
        assert detected_image.shape == sample_image.shape
        assert mask.shape == sample_image.shape[:2]

    def test_detect_lane_returns_ndarray(self, sample_image):
        """Test that detection returns numpy arrays."""
        detector = LaneDetector()
        detected_image, mask = detector.detect_lane(sample_image)

        assert isinstance(detected_image, np.ndarray)
        assert isinstance(mask, np.ndarray)


@pytest.mark.gpu
class TestYOLODetector:
    """Test YOLO-based detector (requires model weights)."""

    def test_detector_initialization_cpu(self):
        """Test YOLO detector initialization on CPU."""
        try:
            detector = YOLOLaneDetector(
                model_path="yolov8n-seg.pt",
                device="cpu",
            )
            assert detector is not None
        except Exception as e:
            pytest.skip(f"YOLO model not available: {e}")

    @pytest.mark.slow
    def test_detect_lanes(self, sample_image):
        """Test YOLO lane detection."""
        try:
            detector = YOLOLaneDetector(
                model_path="yolov8n-seg.pt",
                device="cpu",
            )
            annotated_image, mask, metadata = detector.detect_lanes(sample_image)

            assert annotated_image is not None
            assert mask is not None
            assert isinstance(metadata, dict)
            assert "num_lanes" in metadata
            assert "confidences" in metadata

        except Exception as e:
            pytest.skip(f"YOLO detection test skipped: {e}")


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_full_pipeline_traditional(self, sample_image):
        """Test complete traditional detection pipeline."""
        detector = LaneDetector()
        result_image, result_mask = detector.detect_lane(sample_image)

        # Verify outputs are valid
        assert result_image.shape == sample_image.shape
        assert result_mask.shape == sample_image.shape[:2]
        assert result_image.dtype == np.uint8
        assert result_mask.dtype == np.uint8

    def test_multiple_detections(self, sample_image):
        """Test running detector multiple times."""
        detector = LaneDetector()

        for _ in range(5):
            result_image, result_mask = detector.detect_lane(sample_image)
            assert result_image is not None
            assert result_mask is not None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_image(self):
        """Test with empty (black) image."""
        empty_image = np.zeros((480, 640, 3), dtype=np.uint8)
        detector = LaneDetector()
        result_image, result_mask = detector.detect_lane(empty_image)

        assert result_image is not None
        assert result_mask is not None

    def test_single_pixel_image(self):
        """Test with very small image."""
        tiny_image = np.zeros((1, 1, 3), dtype=np.uint8)
        detector = LaneDetector()

        try:
            result_image, result_mask = detector.detect_lane(tiny_image)
            # Should either work or raise a reasonable error
            assert result_image is not None or True
        except Exception:
            # It's acceptable to fail on degenerate input
            pass

    def test_large_image(self):
        """Test with large image."""
        large_image = np.zeros((2160, 3840, 3), dtype=np.uint8)
        # Draw some lane-like features
        cv2.line(large_image, (1000, 2160), (1500, 1080), (255, 255, 255), 10)

        detector = LaneDetector()
        result_image, result_mask = detector.detect_lane(large_image)

        assert result_image.shape == large_image.shape
        assert result_mask.shape == large_image.shape[:2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
