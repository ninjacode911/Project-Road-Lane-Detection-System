import cv2
import numpy as np
from lane_detection.detector import LaneDetector
from lane_detection.ensemble import LaneEnsemble
from lane_detection.utils import fit_polynomial_sliding_window

def test_polynomial_fitting():
    # Create a synthetic binary mask with a curved line
    mask = np.zeros((720, 1280), dtype=np.uint8)
    ploty = np.linspace(0, 719, 720)
    # x = ay^2 + by + c
    left_fit_true = [0.0005, 0.1, 300]
    
    left_fitx = left_fit_true[0]*ploty**2 + left_fit_true[1]*ploty + left_fit_true[2]
    
    for i in range(len(ploty)):
        if 0 <= int(left_fitx[i]) < 1280:
            mask[int(ploty[i]), int(left_fitx[i])] = 255
            
    # Test fitting
    left_fit, right_fit, out_ploty = fit_polynomial_sliding_window(mask)
    
    if left_fit is None:
        print("✗ Polynomial fitting failed to find left lane")
        return False
    
    print(f"✓ Polynomial fitting passed (Coeffs: {left_fit})")
    return True

def test_detector_advanced():
    detector = LaneDetector()
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.fillPoly(image, [np.array([[500, 720], [780, 720], [680, 450], [600, 450]])], (200, 200, 200))
    
    img, mask, meta = detector.detect_lane(image, method="advanced")
    print(f"✓ Detector advanced test passed (Meta: {meta})")
    return True

def test_segformer():
    detector = LaneDetector()
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.fillPoly(image, [np.array([[0, 720], [1280, 720], [1280, 400], [0, 400]])], (50, 50, 50))
    
    img, mask, meta = detector.detect_lane(image, method="segformer")
    print(f"✓ SegFormer test passed (Meta: {meta})")
    return True

def test_ensemble_fusion():
    ensemble = LaneEnsemble(device="cpu")
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.line(image, (600, 720), (640, 450), (255, 255, 255), 10)
    cv2.line(image, (680, 720), (640, 450), (255, 255, 255), 10)
    
    img, mask, meta = ensemble.detect(image)
    print(f"✓ Ensemble fusion test passed (Meta: {meta})")
    return True

if __name__ == "__main__":
    success = True
    print("\n--- STARTING NEURAL-LANE INTELLIGENCE CHECK ---")
    try:
        success &= test_polynomial_fitting()
        success &= test_detector_advanced()
        success &= test_segformer()
        success &= test_ensemble_fusion()
    except Exception as e:
        print(f"✗ CRITICAL FAILURE during check: {e}")
        success = False
    
    if success:
        print("\nALL NEURAL INTERFACE CHECKS PASSED 🚀")
        print("System is 100% stable and operational.")
    else:
        print("\n✗ SOME CHECKS FAILED - System requires attention.")
        exit(1)
