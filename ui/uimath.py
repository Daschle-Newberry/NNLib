import numpy as np

def bounding_box_check(pos : np.ndarray, bbox : np.ndarray):
    return bbox[0][0] <= pos[0] <= bbox[1][0] and bbox[0][1] <= pos[1] <= bbox[1][1]

def convert_image_to_grayscale(img : np.array):
    return np.dot(img.astype(np.float32), [0.299, 0.587, 0.114]).astype(np.uint8)

def invert_image(img : np.array):
    return 255 - img