"""
Benchmark script for comparing detection methods and measuring performance.
"""

import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from rich.console import Console
from rich.table import Table

from lane_detection import LaneDetector, YOLOLaneDetector

console = Console()


def create_test_images() -> List[np.ndarray]:
    """Create test images of different sizes."""
    sizes = [
        (480, 640),  # VGA
        (720, 1280),  # HD
        (1080, 1920),  # Full HD
    ]

    images = []
    for h, w in sizes:
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Draw some lane-like features
        cv2.line(img, (w//4, h), (w//3, h//2), (255, 255, 255), 5)
        cv2.line(img, (3*w//4, h), (2*w//3, h//2), (255, 255, 255), 5)
        images.append(img)

    return images


def benchmark_method(
    detector,
    images: List[np.ndarray],
    method_name: str,
    num_iterations: int = 10,
) -> Dict[str, float]:
    """Benchmark a detection method."""
    console.print(f"\n[bold cyan]Benchmarking {method_name}...[/bold cyan]")

    results = {}

    for idx, image in enumerate(images):
        h, w = image.shape[:2]
        size_name = f"{w}x{h}"

        times = []
        for _ in range(num_iterations):
            start = time.time()

            if hasattr(detector, "detect_lanes"):
                # YOLO detector
                _, _, _ = detector.detect_lanes(image)
            else:
                # Traditional detector
                _, _ = detector.detect_lane(image)

            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1 / avg_time if avg_time > 0 else 0

        results[size_name] = {
            "avg_time": avg_time,
            "std_time": std_time,
            "fps": fps,
        }

        console.print(
            f"  {size_name}: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms ({fps:.1f} FPS)"
        )

    return results


def main():
    """Run benchmark suite."""
    console.print(
        "\n[bold green]═══════════════════════════════════[/bold green]\n"
        "[bold white]Lane Detection Benchmark Suite[/bold white]\n"
        "[bold green]═══════════════════════════════════[/bold green]\n"
    )

    # Create test images
    console.print("[yellow]Creating test images...[/yellow]")
    images = create_test_images()

    # Initialize detectors
    console.print("\n[yellow]Initializing detectors...[/yellow]")

    try:
        yolo_detector = YOLOLaneDetector(
            model_path="yolov8n-seg.pt",
            device="cuda",
            use_fp16=True,
        )
        have_yolo = True
    except Exception as e:
        console.print(f"[red]Could not load YOLO detector: {e}[/red]")
        have_yolo = False

    traditional_detector = LaneDetector()

    # Run benchmarks
    all_results = {}

    if have_yolo:
        all_results["YOLO"] = benchmark_method(yolo_detector, images, "YOLO")

    all_results["Traditional CV"] = benchmark_method(
        traditional_detector, images, "Traditional CV"
    )

    # Display comparison table
    console.print("\n[bold green]Performance Comparison:[/bold green]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Resolution", style="cyan")
    table.add_column("Method", style="yellow")
    table.add_column("Avg Time (ms)", justify="right")
    table.add_column("Std Dev (ms)", justify="right")
    table.add_column("FPS", justify="right", style="green")

    for size_name in ["640x480", "1280x720", "1920x1080"]:
        first_row = True
        for method_name, method_results in all_results.items():
            if size_name in method_results:
                stats = method_results[size_name]
                table.add_row(
                    size_name if first_row else "",
                    method_name,
                    f"{stats['avg_time']*1000:.2f}",
                    f"{stats['std_time']*1000:.2f}",
                    f"{stats['fps']:.1f}",
                )
                first_row = False

    console.print(table)

    console.print("\n[bold green]Benchmark complete![/bold green]\n")


if __name__ == "__main__":
    main()
