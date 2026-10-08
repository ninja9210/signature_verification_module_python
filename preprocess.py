import os
import cv2
import numpy as np

SIZE = (256, 128)  # (width, height)

def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)          # grayscale
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img

def binarize(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)              # noise removal
    _, th = cv2.threshold(blur, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # signature = white
    return th

def remove_specks(th, min_area=15, rel=0.3):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    if n <= 1:
        return th
    w = stats[1:, cv2.CC_STAT_WIDTH]
    h = stats[1:, cv2.CC_STAT_HEIGHT]
    areas = stats[1:, cv2.CC_STAT_AREA]
    dims = np.maximum(w, h)
    keep = (dims >= rel * dims.max()) & (areas >= min_area)
    ids = np.where(keep)[0] + 1
    return (np.isin(labels, ids) * 255).astype(np.uint8)

def crop(th):
    pts = cv2.findNonZero(th)
    if pts is None:
        raise ValueError("Blank image!")
    x, y, w, h = cv2.boundingRect(pts)
    return th[y:y + h, x:x + w]

def resize_with_padding(img, size=SIZE):
    W, H = size
    h, w = img.shape
    scale = min(W / w, H / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    _, r = cv2.threshold(r, 127, 255, cv2.THRESH_BINARY)
    canvas = np.zeros((H, W), np.uint8)
    x0, y0 = (W - nw) // 2, (H - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return canvas

def preprocess(path, debug_dir=None):
    gray = load_gray(path)
    th = binarize(gray)
    clean = remove_specks(th)
    cropped = crop(clean)
    final = resize_with_padding(cropped)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        for name, im in [("1_gray", gray), ("2_binary", th), ("3_clean", clean),
                         ("4_cropped", cropped), ("5_final", final)]:
            cv2.imwrite(os.path.join(debug_dir, name + ".png"), im)
    return final

if __name__ == "__main__":
    import sys
    preprocess(sys.argv[1], debug_dir="debug_out")