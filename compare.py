import cv2
import numpy as np
from .features import hog_vector, projections, basic_features

WEIGHTS = {"iou": 0.25, "proj": 0.25, "hog": 0.35, "aspect": 0.075, "density": 0.075}

def _ratio_sim(a, b):
    return 1 - abs(a - b) / max(a, b, 1e-9)

def _corr(a, b):
    c = np.corrcoef(a, b)[0, 1]
    return 0.0 if np.isnan(c) else max(float(c), 0.0)

def _cosine(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if d == 0 else float(np.dot(a, b) / d)

def component_scores(a, b):
    k = np.ones((5, 5), np.uint8)             # dilate: thoda sa shift tolerate ho
    ad, bd = cv2.dilate(a, k) > 0, cv2.dilate(b, k) > 0
    iou = np.logical_and(ad, bd).sum() / max(np.logical_or(ad, bd).sum(), 1)

    (ac, ar), (bc, br) = projections(a), projections(b)
    proj = (_corr(ac, bc) + _corr(ar, br)) / 2

    fa, fb = basic_features(a), basic_features(b)
    return {
        "iou": float(iou),
        "proj": proj,
        "hog": _cosine(hog_vector(a), hog_vector(b)),
        "aspect": _ratio_sim(fa["aspect"], fb["aspect"]),
        "density": _ratio_sim(fa["density"], fb["density"]),
    }

def final_score(scores):
    return sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)   # 0 se 1