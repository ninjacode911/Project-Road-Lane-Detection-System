import cv2
import numpy as np

def draw_lane_lines(image, lines, color=(0, 255, 0), thickness=5):
    """
    Draw lane lines on the input image. (Maintains compatibility)
    """
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image

def draw_poly_lane(image, left_fit, right_fit, ploty):
    """
    Draw polynomial fitted lanes on the image.
    """
    out_img = image.copy()
    if left_fit is None or right_fit is None:
        return out_img
        
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(image[:,:,0]).astype(np.uint8)
    color_warp = np.zeros_like(image).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    
    # Draw the lines
    for i in range(len(ploty)-1):
        cv2.line(color_warp, (int(left_fitx[i]), int(ploty[i])), 
                 (int(left_fitx[i+1]), int(ploty[i+1])), (255, 255, 255), 10)
        cv2.line(color_warp, (int(right_fitx[i]), int(ploty[i])), 
                 (int(right_fitx[i+1]), int(ploty[i+1])), (255, 255, 255), 10)

    # Combine the result with the original image
    result = cv2.addWeighted(out_img, 1, color_warp, 0.3, 0)
    return result

def detect_hough_lines(mask):
    """
    Detect lines in a binary mask using Probabilistic Hough Transform.
    """
    edges = cv2.Canny(mask, 50, 150)
    rho = 1
    theta = np.pi/180
    threshold = 20
    min_line_len = 20
    max_line_gap = 300
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def fit_polynomial_sliding_window(binary_warped):
    """
    Fit a second degree polynomial using sliding windows on a binary warped image.
    """
    # Assuming the image is a binary mask of lanes
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    midpoint = int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    margin = 100
    minpix = 50
    window_height = int(binary_warped.shape[0]//nwindows)
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = None
    right_fit = None
    
    if len(leftx) > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) > 0:
        right_fit = np.polyfit(righty, rightx, 2)
        
    return left_fit, right_fit, np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

