import math
import cv2
import numpy as np

def grayscale(img):
    '''Applies the grayscale transform'''
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    '''Applies the Canny transform'''
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    '''Applies a Gaussian Noise Kernel'''
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    '''Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points. '''
    
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255, )*channel_count
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def region_selection(image):
    '''Determine and cut the region of interest in the input image.'''
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        # We could have used fixed numbers as the vertices of the polygon,
        # but they will not be applicable to images with different dimensions.
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def draw_hough_lines(img, lines, color=[255, 0, 0], thickness=2):
    '''Draw hough line'''
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    left_lane_slopes = []
    right_lane_slopes = []
    right_lane_intercepts = []
    left_lane_intercepts = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = calculate_slope(x1, y1, x2, y2)
            if is_positive_slope(slope):
                right_lane_slopes.append(slope)
                right_lane_intercepts.append(calculate_intercept(slope, x1, y1))
            else:
                left_lane_slopes.append(slope)
                left_lane_intercepts.append(calculate_intercept(slope, x1, y1))
                # cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    left_lane_slope = np.mean(left_lane_slopes)
    left_lane_intercept = np.mean(left_lane_intercepts)
    right_lane_slope = np.mean(right_lane_slopes)
    right_lane_intercept = np.mean(right_lane_intercepts)
    
    # get the end points
    y_max = img.shape[0]
    y_min = y_max * 0.6
    
    draw_lane(img, left_lane_slope, left_lane_intercept, y_min, y_max, color, 10)
    draw_lane(img, right_lane_slope, right_lane_intercept, y_min, y_max, color, 10)


def draw_lane(img, slope, intercept, y_min, y_max, color, thickness):
    if not (math.isnan(slope)) and not (slope == float("inf")) and not (slope == float("-inf")):
        x_min = (y_min - intercept) / slope
        x_max = (y_max - intercept) / slope
        cv2.line(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)


def calculate_intercept(slope, x1, y1):
    return y1 - slope * x1


def calculate_slope(x1, y1, x2, y2):
    ''' Calculate the slope of a line given 2 points with
        coordonate(x1, y1), (x2, y2)on the line
    '''
    slope = (y2 - y1) / (x2 - x1)
    return slope


def is_positive_slope(slope):
    return slope > 0


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img



def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

    


        
    
    

