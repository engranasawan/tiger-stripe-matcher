import os
import json
import numpy as np
import cv2
import torch
from PIL import Image

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor


# ----------------------------
# Model manager (cached once)
# ----------------------------
class ModelBundle:
    def __init__(self, sam_ckpt_path: str = "sam_vit_h_4b8939.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo = YOLO("yolov8m.pt")

        if not os.path.exists(sam_ckpt_path):
            raise FileNotFoundError(
                f"SAM checkpoint not found: {sam_ckpt_path}\n"
                "Place it in the project root or update the path."
            )

        sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to(self.device).eval()
        self.sam_predictor = SamPredictor(sam)


def read_image_rgb_from_bytes(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(img)


def read_image_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _largest_cc(mask_u8):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return mask_u8
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = (labels == idx).astype(np.uint8) * 255
    return out


def detect_tiger_box_yolo(bundle: ModelBundle, img_rgb: np.ndarray):
    res = bundle.yolo.predict(img_rgb, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None, 0.0

    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss  = res.boxes.cls.cpu().numpy().astype(int)
    names = res.names

    preferred = set(["cat", "dog", "horse", "cow", "sheep", "bear", "zebra", "giraffe", "elephant"])
    H, W = img_rgb.shape[:2]

    scores = []
    for b, c, k in zip(boxes, confs, clss):
        x1, y1, x2, y2 = b
        area = max(1.0, (x2-x1) * (y2-y1)) / (W*H)
        name = names.get(int(k), "")
        pref = 1.0 if name in preferred else 0.0
        s = (1.6*c) + (1.0*area) + (0.3*pref)
        scores.append(s)

    j = int(np.argmax(scores))
    return boxes[j].astype(np.float32), float(confs[j])


def segment_roi_with_sam(bundle: ModelBundle, img_rgb: np.ndarray, bbox_xyxy: np.ndarray):
    bundle.sam_predictor.set_image(img_rgb)
    masks, scores, _ = bundle.sam_predictor.predict(
        box=bbox_xyxy[None, :],
        multimask_output=True
    )
    m = masks[int(np.argmax(scores))].astype(np.uint8) * 255

    k = max(3, int(round(min(img_rgb.shape[:2]) * 0.01)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
    m = _largest_cc(m)
    return m > 127


def color_constancy_shades_of_gray(img_rgb, power=6):
    img = img_rgb.astype(np.float32) + 1e-6
    norm = np.mean(img ** power, axis=(0,1)) ** (1.0/power)
    img = img / norm
    img = img / (np.max(img) + 1e-6) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def stripe_preprocess(img_rgb, mask, out_size=512):
    img = color_constancy_shades_of_gray(img_rgb)
    bg = np.full_like(img, 127, dtype=np.uint8)
    img = np.where(mask[...,None], img, bg)

    H, W = img.shape[:2]
    scale = out_size / max(H, W)
    nh, nw = int(round(H * scale)), int(round(W * scale))
    img_rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    pad_t = (out_size - nh)//2
    pad_b = out_size - nh - pad_t
    pad_l = (out_size - nw)//2
    pad_r = out_size - nw - pad_l
    img_rs = cv2.copyMakeBorder(img_rs, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(127,127,127))

    m = (mask.astype(np.uint8)*255)
    m = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
    m = cv2.copyMakeBorder(m, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
    m = m > 127

    gray = cv2.cvtColor(img_rs, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    g1 = cv2.GaussianBlur(gray, (0,0), 1.0)
    g2 = cv2.GaussianBlur(gray, (0,0), 3.0)
    dog = cv2.subtract(g1, g2)
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dog = np.where(m, dog, 127).astype(np.uint8)

    edge = cv2.Canny(dog, 60, 140)
    edge = np.where(m, edge, 0).astype(np.uint8)

    return dog, edge, img_rs, m


def keypoint_similarity(edge1, edge2):
    use_sift = hasattr(cv2, "SIFT_create")
    if use_sift:
        det = cv2.SIFT_create(nfeatures=800)
        norm_type = cv2.NORM_L2
    else:
        det = cv2.ORB_create(nfeatures=1200)
        norm_type = cv2.NORM_HAMMING

    k1, d1 = det.detectAndCompute(edge1, None)
    k2, d2 = det.detectAndCompute(edge2, None)

    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return 0.0, 0, 0

    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 8:
        return 0.0, len(good), 0

    pts1 = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    pts2 = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    _, inliers = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    if inliers is None:
        return 0.0, len(good), 0

    inliers = inliers.ravel().astype(bool)
    ninl = int(inliers.sum())
    ratio = ninl / max(1, len(good))
    return float(ratio), int(len(good)), ninl


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compare_tiger_stripes(bundle: ModelBundle, img1_rgb: np.ndarray, img2_rgb: np.ndarray):
    b1, c1 = detect_tiger_box_yolo(bundle, img1_rgb)
    b2, c2 = detect_tiger_box_yolo(bundle, img2_rgb)
    if b1 is None or b2 is None:
        return None, {"label":"negative","confidence":0,"reason":"Tiger box detection failed."}

    m1 = segment_roi_with_sam(bundle, img1_rgb, b1)
    m2 = segment_roi_with_sam(bundle, img2_rgb, b2)

    dog1, edge1, rgb1, _ = stripe_preprocess(img1_rgb, m1)
    dog2, edge2, rgb2, _ = stripe_preprocess(img2_rgb, m2)

    h1 = cv2.calcHist([dog1],[0],None,[64],[0,256]).ravel()
    h2 = cv2.calcHist([dog2],[0],None,[64],[0,256]).ravel()
    h1 = h1 / (h1.sum() + 1e-9)
    h2 = h2 / (h2.sum() + 1e-9)
    sim_hist = float(np.dot(h1, h2))

    inlier_ratio, nmatch, ninl = keypoint_similarity(edge1, edge2)
    det = float(min(c1, c2))

    z = 2.0*(sim_hist - 0.55) + 2.4*(inlier_ratio - 0.25) + 0.3*(det - 0.25)
    p_same = float(sigmoid(z))

    label = "positive" if p_same >= 0.5 else "negative"
    confidence = int(round(p_same*100)) if label=="positive" else int(round((1-p_same)*100))

    result = {
        "label": label,
        "confidence": confidence,
        "details": {
            "p_same": p_same,
            "sim_hist": sim_hist,
            "inlier_ratio": inlier_ratio,
            "matches": nmatch,
            "inliers": ninl,
            "det_conf": det,
        }
    }

    artifacts = {
        "mask1": m1,
        "mask2": m2,
        "dog1": dog1,
        "dog2": dog2,
        "edge1": edge1,
        "edge2": edge2,
        "bbox1": b1,
        "bbox2": b2,
    }

    return artifacts, result
