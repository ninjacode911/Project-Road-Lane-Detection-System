"""
Modern Lane Detection Application - 2026 Edition
Supports multiple detection methods: YOLO, traditional CV, and ensemble.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("results/logs/app.log", rotation="10 MB", level="DEBUG")

# Import detection methods
from lane_detection import (
    AdvancedPreprocessor,
    LaneDetector,
    YOLOLaneDetector,
    detect_hough_lines,
    draw_lane_lines,
)


class ModernLaneDetectionApp:
    """Modern Lane Detection Application with multiple detection methods."""

    def __init__(
        self,
        method: Literal["yolo", "traditional", "ensemble"] = "yolo",
        device: str = "cuda",
    ):
        """
        Initialize the lane detection application.

        Args:
            method: Detection method to use
            device: Device for computation ('cuda', 'cpu', 'mps')
        """
        self.method = method
        self.device = device

        logger.info(f"Initializing Lane Detection App - Method: {method}, Device: {device}")

        # Initialize detectors based on method
        if method == "yolo" or method == "ensemble":
            try:
                self.yolo_detector = YOLOLaneDetector(
                    model_path="yolov8n-seg.pt",
                    device=device,
                    confidence_threshold=0.5,
                )
                logger.info("✓ YOLO detector initialized")
            except Exception as e:
                logger.warning(f"YOLO initialization failed: {e}")
                logger.warning("Falling back to traditional method")
                self.method = "traditional"

        if method == "traditional" or method == "ensemble":
            self.traditional_detector = LaneDetector()
            self.preprocessor = AdvancedPreprocessor()
            logger.info("✓ Traditional detector initialized")

    def process_image(
        self,
        image_path: str,
        output_dir: str = "results/images",
        visualize: bool = True,
    ) -> dict:
        """
        Process a single image for lane detection.

        Args:
            image_path: Path to input image
            output_dir: Directory to save results
            visualize: Whether to save visualization

        Returns:
            Processing statistics
        """
        logger.info(f"Processing image: {image_path}")

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not read image: {image_path}")
            return {"error": "Failed to read image"}

        # Detect lanes based on method
        start_time = time.time()

        if self.method == "yolo":
            detected_image, mask, metadata = self.yolo_detector.detect_lanes(image)
            method_used = "YOLO"
        elif self.method == "traditional":
            detected_image, mask = self.traditional_detector.detect_lane(image)
            metadata = {}
            method_used = "Traditional CV"
        else:  # ensemble
            # Try YOLO first, fallback to traditional
            try:
                detected_image, mask, metadata = self.yolo_detector.detect_lanes(image)
                method_used = "YOLO (Ensemble)"
            except Exception as e:
                logger.warning(f"YOLO failed, using traditional: {e}")
                detected_image, mask = self.traditional_detector.detect_lane(image)
                metadata = {}
                method_used = "Traditional CV (Fallback)"

        processing_time = time.time() - start_time

        # Save results if requested
        if visualize:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_name = Path(image_path).stem

            # Save detected image
            detected_path = output_path / f"{image_name}_detected_{timestamp}.jpg"
            cv2.imwrite(str(detected_path), detected_image)

            # Save mask
            mask_path = output_path / f"{image_name}_mask_{timestamp}.jpg"
            cv2.imwrite(str(mask_path), mask)

            logger.info(f"✓ Results saved: {detected_path}")

        # Return statistics
        stats = {
            "method": method_used,
            "processing_time": processing_time,
            "fps": 1 / processing_time if processing_time > 0 else 0,
            "image_shape": image.shape,
            **metadata,
        }

        logger.info(f"Processing complete: {processing_time:.3f}s ({stats['fps']:.1f} FPS)")

        return stats

    def process_video(
        self,
        video_path: str,
        output_dir: str = "results/videos",
        skip_frames: int = 0,
        show_preview: bool = False,
    ) -> dict:
        """
        Process a video for lane detection.

        Args:
            video_path: Path to input video
            output_dir: Directory to save results
            skip_frames: Process every Nth frame (0 = all frames)
            show_preview: Show real-time preview window

        Returns:
            Processing statistics
        """
        logger.info(f"Processing video: {video_path}")

        # Use YOLO's built-in video processing if available
        if self.method == "yolo":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(output_dir) / f"detected_{timestamp}.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            stats = self.yolo_detector.process_video(
                str(video_path),
                str(output_path),
                skip_frames=skip_frames,
                show_preview=show_preview,
            )
            logger.info(f"✓ Video saved: {output_path}")
            return stats

        # Traditional video processing
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return {"error": "Failed to open video"}

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Prepare output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / f"detected_{timestamp}.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Process frames
        frame_idx = 0
        processed_frames = 0
        start_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames if requested
                if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                    frame_idx += 1
                    continue

                # Detect lanes
                detected_frame, _ = self.traditional_detector.detect_lane(frame)
                out.write(detected_frame)

                # Show preview if requested
                if show_preview:
                    cv2.imshow("Lane Detection", detected_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                processed_frames += 1
                frame_idx += 1

                # Log progress every 100 frames
                if processed_frames % 100 == 0:
                    logger.info(f"Processed {processed_frames}/{total_frames} frames")

        finally:
            cap.release()
            out.release()
            if show_preview:
                cv2.destroyAllWindows()

        processing_time = time.time() - start_time
        avg_fps = processed_frames / processing_time if processing_time > 0 else 0

        stats = {
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "processing_time": processing_time,
            "average_fps": avg_fps,
            "output_path": str(output_path),
        }

        logger.info(f"✓ Video processing complete: {processing_time:.2f}s ({avg_fps:.1f} FPS)")
        logger.info(f"✓ Output saved: {output_path}")

        return stats


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Modern Lane Detection System - 2026 Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to input image or video",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["yolo", "traditional", "ensemble"],
        default="yolo",
        help="Detection method to use (default: yolo)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "mps"],
        default="cuda",
        help="Device to use for computation (default: cuda)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/images or results/videos)",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="For videos: process every Nth frame (default: 0, process all)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show real-time preview (videos only)",
    )

    args = parser.parse_args()

    # Initialize app
    app = ModernLaneDetectionApp(method=args.method, device=args.device)

    # Determine if input is image or video
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Check file type
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    is_video = input_path.suffix.lower() in video_extensions

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = "results/videos" if is_video else "results/images"

    # Process
    if is_video:
        stats = app.process_video(
            str(input_path),
            output_dir=output_dir,
            skip_frames=args.skip_frames,
            show_preview=args.preview,
        )
    else:
        stats = app.process_image(
            str(input_path),
            output_dir=output_dir,
        )

    # Print statistics
    logger.info("=" * 50)
    logger.info("Processing Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
