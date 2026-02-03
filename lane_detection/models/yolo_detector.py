"""
YOLOv8-based Lane Detection Module
Modern real-time lane detection using Ultralytics YOLOv8 segmentation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results


class YOLOLaneDetector:
    """
    YOLOv8-based lane detector for real-time lane segmentation.
    
    Features:
    - Real-time processing (60+ FPS)
    - Instance segmentation for individual lanes
    - Confidence scoring
    - GPU acceleration support
    """

    def __init__(
        self,
        model_path: str = "yolov8n-seg.pt",
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        use_fp16: bool = True,
    ):
        """
        Initialize YOLO lane detector.

        Args:
            model_path: Path to YOLO model weights
            device: Device to run on ('cuda', 'cpu', 'mps')
            confidence_threshold: Minimum confidence for detections
            iou_threshold: NMS IoU threshold
            use_fp16: Use FP16 precision for faster inference
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.use_fp16 = use_fp16

        # Check device availability
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = "cpu"

        # Load YOLO model
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> YOLO:
        """Load and configure YOLO model."""
        model = YOLO(model_path)
        
        # Move model to device
        model.to(self.device)
        
        # Enable FP16 if requested and supported
        if self.use_fp16 and self.device == "cuda":
            model.model.half()
        
        return model

    def detect_lanes(
        self,
        image: np.ndarray,
        image_size: int = 640,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Detect lanes in an image using YOLO segmentation.

        Args:
            image: Input image (BGR format)
            image_size: Input size for YOLO model

        Returns:
            Tuple of:
            - annotated_image: Image with lane overlays
            - mask: Binary mask of lane regions
            - metadata: Detection metadata (confidence, boxes, etc.)
        """
        # Run inference
        results: List[Results] = self.model.predict(
            image,
            imgsz=image_size,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        # Extract first result
        result = results[0]

        # Create binary mask
        mask = self._create_mask(result, image.shape[:2])

        # Annotate image
        annotated_image = self._annotate_image(image.copy(), result)

        # Extract metadata
        metadata = self._extract_metadata(result)

        return annotated_image, mask, metadata

    def _create_mask(self, result: Results, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create binary mask from segmentation results.

        Args:
            result: YOLO results object
            image_shape: (height, width) of original image

        Returns:
            Binary mask (0 or 255)
        """
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # Check if masks exist
        if result.masks is None or len(result.masks) == 0:
            return mask

        # Combine all lane masks
        for seg_mask in result.masks.data:
            # Resize mask to image size
            seg_mask_resized = cv2.resize(
                seg_mask.cpu().numpy(),
                (w, h),
                interpolation=cv2.INTER_LINEAR,
            )
            # Threshold and add to combined mask
            mask = np.maximum(mask, (seg_mask_resized > 0.5).astype(np.uint8) * 255)

        return mask

    def _annotate_image(self, image: np.ndarray, result: Results) -> np.ndarray:
        """
        Draw lane annotations on image.

        Args:
            image: Input image
            result: YOLO results

        Returns:
            Annotated image
        """
        # Use YOLO's built-in plotting
        annotated = result.plot(
            conf=True,
            boxes=False,  # Don't show bounding boxes for lanes
            labels=False,  # Don't show class labels
        )

        return annotated

    def _extract_metadata(self, result: Results) -> Dict[str, Any]:
        """
        Extract detection metadata.

        Args:
            result: YOLO results

        Returns:
            Dictionary with metadata
        """
        metadata = {
            "num_lanes": len(result.boxes) if result.boxes is not None else 0,
            "confidences": [],
            "boxes": [],
            "has_segmentation": result.masks is not None,
        }

        if result.boxes is not None and len(result.boxes) > 0:
            metadata["confidences"] = result.boxes.conf.cpu().numpy().tolist()
            metadata["boxes"] = result.boxes.xyxy.cpu().numpy().tolist()

        return metadata

    def process_video(
        self,
        video_path: str,
        output_path: str,
        skip_frames: int = 0,
        show_preview: bool = False,
    ) -> Dict[str, Any]:
        """
        Process entire video for lane detection.

        Args:
            video_path: Path to input video
            output_path: Path to save output video
            skip_frames: Process every Nth frame (0 = all frames)
            show_preview: Show real-time preview

        Returns:
            Processing statistics
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Processing statistics
        stats = {
            "total_frames": total_frames,
            "processed_frames": 0,
            "skipped_frames": 0,
            "average_confidence": 0.0,
        }

        frame_idx = 0
        total_confidence = 0.0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames if requested
                if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                    stats["skipped_frames"] += 1
                    frame_idx += 1
                    continue

                # Detect lanes
                annotated_frame, _, metadata = self.detect_lanes(frame)

                # Update statistics
                stats["processed_frames"] += 1
                if metadata["confidences"]:
                    total_confidence += np.mean(metadata["confidences"])

                # Write frame
                out.write(annotated_frame)

                # Show preview if requested
                if show_preview:
                    cv2.imshow("Lane Detection", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_idx += 1

        finally:
            cap.release()
            out.release()
            if show_preview:
                cv2.destroyAllWindows()

        # Calculate average confidence
        if stats["processed_frames"] > 0:
            stats["average_confidence"] = total_confidence / stats["processed_frames"]

        return stats


# Convenience function for quick lane detection
def detect_lanes_yolo(
    image: np.ndarray,
    model_path: str = "yolov8n-seg.pt",
    confidence: float = 0.5,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quick lane detection using YOLO.

    Args:
        image: Input image
        model_path: Path to YOLO model
        confidence: Confidence threshold
        device: Device to use

    Returns:
        (annotated_image, mask)
    """
    detector = YOLOLaneDetector(
        model_path=model_path,
        device=device,
        confidence_threshold=confidence,
    )
    annotated_image, mask, _ = detector.detect_lanes(image)
    return annotated_image, mask
