import cv2
import numpy as np
from .compare import component_scores

def centroid_align(img):
    m = cv2.moments(img, binaryImage=True)
    if m["m00"] == 0:
        return img
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    h, w = img.shape
    M = np.float32([[1, 0, w / 2 - cx], [0, 1, h / 2 - cy]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST)

def grid_vec(img):
    return (img.reshape(8, 16, 16, 16).mean(axis=(1, 3)) / 255.0).flatten()

def cosine(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if d == 0 else float(np.dot(a, b) / d)

def chamfer_sim(a, b):
    da = cv2.distanceTransform(255 - a, cv2.DIST_L2, 3)
    db = cv2.distanceTransform(255 - b, cv2.DIST_L2, 3)
    d1 = db[a > 0].mean() if (a > 0).any() else 99
    d2 = da[b > 0].mean() if (b > 0).any() else 99
    return float(np.exp(-(d1 + d2) / 2 / 8.0))

KEYS = ["iou", "proj", "hog", "grid", "chamfer"]

def all_scores(a, b):
    c = component_scores(a, b)
    return {"iou": c["iou"], "proj": c["proj"], "hog": c["hog"],
            "grid": cosine(grid_vec(a), grid_vec(b)),
            "chamfer": chamfer_sim(a, b)}