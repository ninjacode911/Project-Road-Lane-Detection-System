"""
FastAPI Web Server for Lane Detection System
Provides REST API and WebSocket streaming for real-time lane detection.
"""

import asyncio
import io
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from loguru import logger
from pydantic import BaseModel

# Import lane detection modules
from lane_detection import YOLOLaneDetector
from lane_detection.detector import LaneDetector
from lane_detection.ensemble import LaneEnsemble

# Configuration
UPLOAD_DIR = Path("data/uploads")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models/weights")

# Create directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "images").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "videos").mkdir(parents=True, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events for the FastAPI application."""
    logger.info("Starting Lane Detection API v2.0.0")
    logger.info("Detectors will be loaded lazily on first request")
    yield
    logger.info("Shutting down Lane Detection API")

# Initialize FastAPI app
app = FastAPI(
    title="Lane Detection API",
    description="Modern lane detection system with YOLOv8",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file mounting
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Global LaneDetector instance (unified)
lane_detector: Optional[LaneDetector] = None

def get_lane_detector() -> LaneDetector:
    """Get or create the unified lane detector instance."""
    global lane_detector
    if lane_detector is None:
        logger.info("Initializing Unified Lane Detector...")
        lane_detector = LaneDetector(device="cpu")
        logger.info("✓ Lane Detector initialized with all backends")
    return lane_detector


# Removed individual getters in favor of get_lane_detector


# ============================================================================
# Pydantic Models
# ============================================================================


class DetectionResponse(BaseModel):
    """Response model for detection results."""

    success: bool
    processing_time: float
    fps: float
    num_lanes: int
    confidences: List[float]
    curvature: float = 0
    offset: float = 0
    warning: str = "None"
    image_url: Optional[str] = None
    mask_url: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    detector_loaded: bool


# ============================================================================
# API Routes
# ============================================================================


@app.get("/")
async def root():
    """Serve the web interface."""
    html_file = Path(__file__).parent / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    else:
        # Fallback to JSON if HTML not found
        return {
            "name": "Lane Detection API",
            "version": "2.0.0",
            "docs": "/docs",
            "health": "/health",
            "error": "HTML interface not found",
        }


@app.get("/api/info", response_model=dict)
async def api_info():
    """API information endpoint (moved from root)."""
    return {
        "name": "Lane Detection API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        detector_loaded=lane_detector is not None,
    )


@app.post("/detect/image", response_model=DetectionResponse)
async def detect_image(file: UploadFile = File(...), method: str = "yolo"):
    """
    Detect lanes in an uploaded image.
    """
    try:
        if not file.content_type or not file.content_type.startswith("image"):
            raise HTTPException(status_code=400, detail="File must be an image")

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        start_time = time.time()
        det = get_lane_detector()
        annotated_image, mask, metadata = det.detect_lane(image, method=method)

        processing_time = time.time() - start_time

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"

        detected_path = RESULTS_DIR / "images" / f"detected_{filename}"
        mask_path = RESULTS_DIR / "images" / f"mask_{filename}"

        cv2.imwrite(str(detected_path), annotated_image)
        cv2.imwrite(str(mask_path), mask)

        return DetectionResponse(
            success=True,
            processing_time=processing_time,
            fps=1 / processing_time if processing_time > 0 else 0,
            num_lanes=metadata.get("num_lanes", 0),
            confidences=metadata.get("confidences", []),
            curvature=metadata.get("curvature", 0),
            offset=metadata.get("offset", 0),
            warning=metadata.get("warning", "None"),
            image_url=f"/results/images/detected_{filename}",
            mask_url=f"/results/images/mask_{filename}",
        )

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    """
    Detect lanes in an uploaded video.

    Args:
        file: Uploaded video file

    Returns:
        Job ID and status endpoint
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("video"):
            raise HTTPException(status_code=400, detail="File must be a video")

        # Save uploaded video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_path = UPLOAD_DIR / f"{timestamp}_{file.filename}"

        with open(input_path, "wb") as f:
            contents = await file.read()
            f.write(contents)

        # Process video (this will be async in production)
        output_path = RESULTS_DIR / "videos" / f"detected_{timestamp}.mp4"
        det = get_lane_detector()
        
        # Support traditional video processing via YOLO detector's internal logic 
        # or simplified fallback
        from lane_detection.models.yolo_detector import YOLOLaneDetector
        yolo_det = YOLOLaneDetector(device="cpu") # Temporary for video batch

        stats = det.process_video(
            str(input_path),
            str(output_path),
            skip_frames=0,
            show_preview=False,
        )

        # Clean up input file
        input_path.unlink()

        return {
            "success": True,
            "video_url": f"/results/videos/detected_{timestamp}.mp4",
            "stats": stats,
        }

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results/{result_type}/{filename}")
async def get_result_file(result_type: str, filename: str):
    """
    Get a result file (image or video).

    Args:
        result_type: Type of result ('images' or 'videos')
        filename: Name of the file

    Returns:
        File content
    """
    file_path = RESULTS_DIR / result_type / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path)


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video stream processing.
    
    Client should send frames as base64-encoded JPEG images.
    Server responds with processed frames in the same format.
    """
    await websocket.accept()
    logger.info("WebSocket connection established")

    det = get_lane_detector()

    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_bytes()

            # Decode frame
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_json({"error": "Could not decode frame"})
                continue

            # Process frame
            start_time = time.time()
            det = get_lane_detector()
            annotated_frame, _, metadata = det.detect_lane(frame, method="ensemble") # Default WS to ensemble
            processing_time = time.time() - start_time

            # Encode result
            _, buffer = cv2.imencode(".jpg", annotated_frame)
            result_bytes = buffer.tobytes()

            # Send back to client
            await websocket.send_bytes(result_bytes)

            # Also send metadata
            await websocket.send_json(
                {
                    "processing_time": processing_time,
                    "fps": 1 / processing_time if processing_time > 0 else 0,
                    **metadata,
                }
            )

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("WebSocket connection closed")


@app.delete("/results/{result_type}/{filename}")
async def delete_result(result_type: str, filename: str):
    """Delete a result file."""
    file_path = RESULTS_DIR / result_type / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    file_path.unlink()
    return {"success": True, "message": f"Deleted {filename}"}


@app.get("/results/list")
async def list_results():
    """List all result files."""
    results = {
        "images": [],
        "videos": [],
    }

    # List images
    images_dir = RESULTS_DIR / "images"
    if images_dir.exists():
        results["images"] = [
            {
                "name": f.name,
                "url": f"/results/images/{f.name}",
                "size": f.stat().st_size,
                "created": datetime.fromtimestamp(f.stat().st_ctime).isoformat(),
            }
            for f in images_dir.glob("*")
            if f.is_file() and f.suffix.lower() in {".jpg", ".png"}
        ]

    # List videos
    videos_dir = RESULTS_DIR / "videos"
    if videos_dir.exists():
        results["videos"] = [
            {
                "name": f.name,
                "url": f"/results/videos/{f.name}",
                "size": f.stat().st_size,
                "created": datetime.fromtimestamp(f.stat().st_ctime).isoformat(),
            }
            for f in videos_dir.glob("*")
            if f.is_file() and f.suffix.lower() in {".mp4", ".avi"}
        ]

    return results


# ============================================================================
# Startup/Shutdown Events
# ============================================================================


# Removed deprecated on_event handlers in favor of lifespan


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
