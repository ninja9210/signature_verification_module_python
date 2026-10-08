import cv2
import numpy as np

# winSize (256,128) = preprocess.SIZE se match karna zaroori hai
_hog = cv2.HOGDescriptor((256, 128), (32, 32), (16, 16), (16, 16), 9)

def hog_vector(img):
    return _hog.compute(img).flatten()

def projections(img):
    return img.sum(axis=0) / 255.0, img.sum(axis=1) / 255.0   # column, row profiles

def basic_features(img):
    pts = cv2.findNonZero(img)
    _, _, w, h = cv2.boundingRect(pts)
    n_components, _ = cv2.connectedComponents(img)
    return {
        "aspect": w / h,                      # padding ki wajah se aspect ratio preserved hai
        "density": float((img > 0).mean()),   # kitna hissa ink hai
        "components": n_components - 1,
    }