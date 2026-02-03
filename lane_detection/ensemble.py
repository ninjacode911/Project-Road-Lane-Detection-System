import cv2
import numpy as np

class LaneEnsemble:
    def __init__(self, yolo_model_path="yolov8n-seg.pt", device="cpu"):
        self.yolo_model_path = yolo_model_path
        self.device = device
        self.yolo_detector = None
        self.segformer_detector = None
        self.traditional_detector = None

    def _init_models(self):
        if self.traditional_detector is None:
            from .detector import LaneDetector
            from .models.yolo_detector import YOLOLaneDetector
            from .models.segformer import SegFormerLaneDetector
            
            self.traditional_detector = LaneDetector(device=self.device)
            self.yolo_detector = YOLOLaneDetector(model_path=self.yolo_model_path, device=self.device)
            self.segformer_detector = SegFormerLaneDetector(device=self.device)

    def detect(self, image, yolo_weight=0.5, seg_weight=0.3, cv_weight=0.2):
        """
        Ensemble detection combining YOLO, SegFormer, and Traditional CV.
        """
        self._init_models()
        # 1. Get YOLO results
        yolo_annotated, yolo_mask, yolo_meta = self.yolo_detector.detect_lanes(image)
        
        # 2. Get SegFormer results
        _, seg_mask, seg_meta = self.segformer_detector.detect_lanes(image)
        
        # 3. Get Traditional results
        _, cv_mask, cv_meta = self.traditional_detector.detect_lane(image, method="basic")
        
        # 4. Fuse Masks (Weighted combination)
        fused_mask = (yolo_mask.astype(float) * yolo_weight + 
                      seg_mask.astype(float) * seg_weight + 
                      cv_mask.astype(float) * cv_weight)
        
        _, fused_binary = cv2.threshold(fused_mask.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
        
        # 5. Advanced Fitting
        from .utils import fit_polynomial_sliding_window, draw_poly_lane
        left_fit, right_fit, ploty = fit_polynomial_sliding_window(fused_binary)
        
        metadata = {
            "method": "ensemble",
            "yolo_conf": yolo_meta.get("confidences", []),
            "segformer_conf": seg_meta.get("confidence", 0),
            "num_lanes": yolo_meta.get("num_lanes", 0),
            "curvature": 0,
            "offset": 0,
            "warning": "None"
        }

        if left_fit is not None and right_fit is not None:
            detected_image = draw_poly_lane(image, left_fit, right_fit, ploty)
            curvature, offset = self.traditional_detector._calculate_metrics(left_fit, right_fit, image.shape)
            metadata["curvature"] = curvature
            metadata["offset"] = offset
            if abs(offset) > 0.5:
                metadata["warning"] = "Departure Warning: Left" if offset < 0 else "Departure Warning: Right"
        else:
            # Fallback to YOLO annotation if fitting fails
            detected_image = yolo_annotated

        return detected_image, fused_binary, metadata
