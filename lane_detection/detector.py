from .preprocessor import preprocess_image
from .utils import draw_lane_lines, detect_hough_lines, fit_polynomial_sliding_window, draw_poly_lane
import numpy as np
import cv2

class LaneDetector:
    def __init__(self, device="cpu"):
        """Initialize the LaneDetector with multiple backends."""
        self.left_fit = None
        self.right_fit = None
        self.xm_per_pix = 3.7/700
        self.ym_per_pix = 30/720
        self.device = device
        
        # Temporal smoothing history
        self.fit_history = {"left": [], "right": []}
        self.smoothing_factor = 0.8 # New weight (0.2 old, 0.8 new)
        
        # Lazy initialization of models to save memory
        self.yolo = None
        self.segformer = None
        self.ensemble = None

    def detect_lane(self, image, method="advanced"):
        """
        Detect lanes in the input image.
        Methods: "advanced" (CV), "yolo", "segformer", "ensemble".
        """
        try:
            metadata = {"num_lanes": 0, "curvature": 0, "offset": 0, "warning": "None", "method": method}
            
            if method == "yolo":
                if self.yolo is None:
                    from .models.yolo_detector import YOLOLaneDetector
                    self.yolo = YOLOLaneDetector(device=self.device)
                annotated, mask, meta = self.yolo.detect_lanes(image)
                metadata.update(meta)
                return annotated, mask, metadata
                
            elif method == "segformer":
                if self.segformer is None:
                    from .models.segformer import SegFormerLaneDetector
                    self.segformer = SegFormerLaneDetector(device=self.device)
                annotated, mask, meta = self.segformer.detect_lanes(image)
                metadata.update(meta)
                return annotated, mask, metadata
                
            elif method == "ensemble":
                if self.ensemble is None:
                    from .ensemble import LaneEnsemble
                    self.ensemble = LaneEnsemble(device=self.device)
                annotated, mask, meta = self.ensemble.detect(image)
                metadata.update(meta)
                return annotated, mask, metadata

            # Default: Advanced Traditional CV
            processed_image, mask = preprocess_image(image)
            left_fit, right_fit, ploty = fit_polynomial_sliding_window(mask)
            
            if left_fit is not None and right_fit is not None:
                # Apply Temporal Smoothing
                if self.left_fit is not None:
                    left_fit = left_fit * self.smoothing_factor + self.left_fit * (1 - self.smoothing_factor)
                    right_fit = right_fit * self.smoothing_factor + self.right_fit * (1 - self.smoothing_factor)
                
                self.left_fit = left_fit
                self.right_fit = right_fit
                
                detected_image = draw_poly_lane(image, left_fit, right_fit, ploty)
                curvature, offset = self._calculate_metrics(left_fit, right_fit, image.shape)
                metadata.update({"curvature": curvature, "offset": offset, "num_lanes": 2})
                if abs(offset) > 0.5:
                    metadata["warning"] = "Departure Warning: Left" if offset < 0 else "Departure Warning: Right"
                return detected_image, mask, metadata
            
            # Fallback to Basic
            lines = detect_hough_lines(mask)
            lane_image = np.zeros_like(image)
            if lines is not None:
                lane_image = draw_lane_lines(lane_image, lines)
                metadata["num_lanes"] = len(lines)
            return cv2.addWeighted(image, 0.8, lane_image, 1.0, 0), mask, metadata

        except Exception as e:
            print(f"Error during lane detection: {e}")
            return image, np.zeros(image.shape[:2], dtype=np.uint8), {"error": str(e)}

    def _calculate_metrics(self, left_fit, right_fit, img_shape):
        """Calculate lane curvature and vehicle offset."""
        h = img_shape[0]
        y_eval = h - 1
        
        # Calculate real-world curvature
        left_curverad = ((1 + (2*left_fit[0]*y_eval*self.ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*y_eval*self.ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
        curvature = (left_curverad + right_curverad) / 2
        
        # Calculate vehicle offset
        left_x = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
        right_x = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
        lane_center = (left_x + right_x) / 2
        vehicle_center = img_shape[1] / 2
        offset = (vehicle_center - lane_center) * self.xm_per_pix
        
        return curvature, offset
