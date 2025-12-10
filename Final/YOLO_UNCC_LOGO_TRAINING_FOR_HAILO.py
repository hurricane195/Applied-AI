
"""
YOLOv8n Training
Produced .onnex file to be converted to .hef
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, List
import argparse  # kept in case you want to extend later, but not required now
import random
import shutil
import os
import stat
import time

import cv2
import numpy as np
import yaml
from tqdm import tqdm
import albumentations as A
from ultralytics import YOLO
import torch  # for CUDA/CPU detection

# -------------------------------------------------------------------------
# HCONFIG FOR UNCC LOGO DETECTion
# -------------------------------------------------------------------------
from pathlib import Path  # make sure this is present near the top

POS_DIR  = Path(r"C:....")
NEG_DIR  = Path(r"C:....")
WORK_DIR = Path(r"C:.....")
RUN_NAME = "MWVPi_IDvsUNCCvsLogoIDsHouse"


IMG_SIZE    = 320
EPOCHS      = 120
BATCH       = 16
TRAIN_COUNT = 1800
VAL_COUNT   = 200
DEVICE      = "0"
MODEL_NAME  = "yolov8n.pt"
POS_FRACTION = 0.7

DEFAULT_CLASS_NAMES = ["target_logo"]
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# --------------------------- Utility / IO helpers -----------------------------

def list_images(folder: Path) -> List[Path]:
    return [p for p in folder.rglob('*') if p.suffix.lower() in EXTS]

def save_yolo_label(label_path: Path, cls: int, x: float, y: float, w: float, h: float):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, 'w') as f:
        f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

# ---- Safer rmtree (avoids Windows readonly/lock issues) ----

def _make_writable_and_retry(func, path, exc):
    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass
    try:
        func(path)
    except PermissionError:
        time.sleep(0.3)
        func(path)

def safe_rmtree(path: Path):
    if not path.exists():
        return
    try:
        shutil.rmtree(path, onerror=_make_writable_and_retry)
    except PermissionError:
        time.sleep(0.5)
        shutil.rmtree(path, onerror=_make_writable_and_retry)

# --------------------------- Synthesis / Compositing --------------------------

def load_image_rgba(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load image; if it has alpha, return (bgr, alpha). Else full-opaque mask."""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise RuntimeError(f"Failed to read {path}")
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
    if im.shape[2] == 3:
        bgr = im
        a = np.full(bgr.shape[:2], 255, dtype=np.uint8)
    else:
        bgr = im[..., :3]
        a = im[..., 3]
    return bgr, a

def rand_resize_and_rotate(bgr: np.ndarray, a: np.ndarray,
                           scale_range=(0.18, 0.45), angle_range=(-12, 12)):
    h, w = bgr.shape[:2]
    s = random.uniform(*scale_range)
    nh, nw = max(1, int(h * s)), max(1, int(w * s))
    bgr_s = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    a_s   = cv2.resize(a,   (nw, nh), interpolation=cv2.INTER_NEAREST)
    ang = random.uniform(*angle_range)
    M = cv2.getRotationMatrix2D((nw/2, nh/2), ang, 1.0)
    bgr_r = cv2.warpAffine(bgr_s, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0,0,0))
    a_r   = cv2.warpAffine(a_s,   M, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0)
    return bgr_r, a_r

def paste_rgba(bg: np.ndarray, fg: np.ndarray, a: np.ndarray, x: int, y: int):
    H, W = bg.shape[:2]
    h, w = fg.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x1 >= x2 or y1 >= y2:
        return None
    roi = bg[y1:y2, x1:x2]
    fx1, fy1 = x1 - x, y1 - y
    fx2, fy2 = fx1 + (x2 - x1), fy1 + (y2 - y1)
    fgc = fg[fy1:fy2, fx1:fx2]
    ac  = a[fy1:fy2, fx1:fx2]
    alpha = (ac.astype(np.float32) / 255.0)[:, :, None]
    out = alpha * fgc + (1.0 - alpha) * roi
    bg[y1:y2, x1:x2] = out
    return (x1, y1, x2, y2)

AUG_BG = A.Compose([
    A.RandomBrightnessContrast(p=0.35, brightness_limit=0.25, contrast_limit=0.25),
    A.HueSaturationValue(p=0.25, hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=15),
    A.MotionBlur(p=0.15, blur_limit=5),
    A.GaussNoise(p=0.15, var_limit=(5.0, 20.0)),
])

AUG_FG = A.Compose([
    A.RandomBrightnessContrast(p=0.35, brightness_limit=0.2, contrast_limit=0.2),
    A.RGBShift(p=0.20, r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
    A.CLAHE(p=0.15, clip_limit=2.0),
])

def make_square_bg(bg_img: np.ndarray, out_size: int) -> np.ndarray:
    """Take a background image and embed it into an out_size × out_size canvas."""
    H, W = bg_img.shape[:2]
    aug_bg = AUG_BG(image=bg_img)['image']
    scale = min(out_size / H, out_size / W)
    nh, nw = int(round(H * scale)), int(round(W * scale))
    canvas = np.full((out_size, out_size, 3), 114, dtype=np.uint8)
    top  = (out_size - nh) // 2
    left = (out_size - nw) // 2
    resized = cv2.resize(aug_bg, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas[top:top+nh, left:left+nw] = resized
    return canvas

def synth_one(bg_img: np.ndarray, logo_bgr: np.ndarray, logo_a: np.ndarray, out_size: int):
    """
    Compose a single positive example:
    - Square-resized background
    - target_logo pasted on top
    Returns (composite_image, bbox_in_pixels) or (image, None) on failure.
    """
    H, W = bg_img.shape[:2]
    aug_bg = AUG_BG(image=bg_img)['image']
    scale = min(out_size / H, out_size / W)
    nh, nw = int(round(H * scale)), int(round(W * scale))
    canvas = np.full((out_size, out_size, 3), 114, dtype=np.uint8)
    top  = (out_size - nh) // 2
    left = (out_size - nw) // 2
    resized = cv2.resize(aug_bg, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas[top:top+nh, left:left+nw] = resized

    lbgr = AUG_FG(image=logo_bgr)['image']
    lbgr, la = rand_resize_and_rotate(lbgr, logo_a)

    h, w = lbgr.shape[:2]
    x = random.randint(left, max(left, left+nw - w)) if w < nw else left
    y = random.randint(top,  max(top,  top+nh - h)) if h < nh else top
    bbox = paste_rgba(canvas, lbgr, la, x, y)
    if bbox is None:
        x = left + (nw - w)//2
        y = top  + (nh - h)//2
        bbox = paste_rgba(canvas, lbgr, la, x, y)
    return canvas, bbox

def bbox_to_yolo(bbox, img_w, img_h):
    x1,y1,x2,y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw/2
    cy = y1 + bh/2
    return cx/img_w, cy/img_h, bw/img_w, bh/img_h

# ------------------------------- Dataset builder ------------------------------

def build_dataset(pos_dir: Path, neg_dir: Path, work_dir: Path,
                  imgsz: int, n_train: int, n_val: int,
                  pos_fraction: float = 0.7):
    """
    Build synthetic dataset.

    pos_fraction: fraction of synthesized images that contain the target_logo.
                  Remaining images are pure background (true negatives).
    """
    ds = work_dir / "dataset"
    img_train = ds / "images" / "train"
    img_val   = ds / "images" / "val"
    lbl_train = ds / "labels" / "train"
    lbl_val   = ds / "labels" / "val"
    for d in [img_train, img_val, lbl_train, lbl_val]:
        d.mkdir(parents=True, exist_ok=True)

    pos_list = list_images(pos_dir)
    neg_list = list_images(neg_dir)
    if not pos_list:
        raise RuntimeError(f"No positive (logo) images found in {pos_dir}")
    if not neg_list:
        raise RuntimeError(f"No negative/background images found in {neg_dir}")

    random.shuffle(pos_list)
    random.shuffle(neg_list)

    calib_candidates: List[Path] = []

    def synth_split(count, img_out_dir, lbl_out_dir):
        nonlocal calib_candidates
        used_neg = 0

        for i in tqdm(range(count), desc=f"Synth {img_out_dir.name}"):
            # Always choose a background from NEG folder
            bg_path = neg_list[used_neg % len(neg_list)]
            used_neg += 1
            bg = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
            if bg is None:
                continue

            # Decide if this image will contain the logo or be pure background
            has_logo = (random.random() < pos_fraction)

            if has_logo:
                # POSITIVE: paste logo onto background
                logo_path = pos_list[i % len(pos_list)]
                try:
                    lbgr, la = load_image_rgba(logo_path)
                    comp, bbox = synth_one(bg, lbgr, la, imgsz)
                    if bbox is None:
                        continue
                except Exception as e:
                    print("[warn] positive synth failed:", e)
                    continue
            else:
                # NEGATIVE: background only, no label
                try:
                    comp = make_square_bg(bg, imgsz)
                    bbox = None
                except Exception as e:
                    print("[warn] negative synth failed:", e)
                    continue

            stem = f"{img_out_dir.name}_{i:06d}"
            img_path = img_out_dir / f"{stem}.jpg"
            lab_path = lbl_out_dir / f"{stem}.txt"

            cv2.imwrite(str(img_path), comp, [cv2.IMWRITE_JPEG_QUALITY, 92])

            if bbox is not None:
                x, y, w, h = bbox_to_yolo(bbox, comp.shape[1], comp.shape[0])
                save_yolo_label(lab_path, cls=0, x=x, y=y, w=w, h=h)

                if img_out_dir.name == 'train' and random.random() < 0.15:
                    calib_candidates.append(img_path)
            # pure background -> no label

    # Build train/val splits
    synth_split(n_train, img_train, lbl_train)
    synth_split(n_val,   img_val,   lbl_val)

    # data.yaml for Ultralytics
    data_yaml = {
        'path': str(ds).replace('\\', '/'),
        'train': 'images/train',
        'val':   'images/val',
        'names': DEFAULT_CLASS_NAMES,
        'nc': len(DEFAULT_CLASS_NAMES)
    }
    with open(ds / 'data.yaml', 'w') as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    # Calibration list file (absolute paths)
    calib_list = ds / 'calib_list.txt'
    calib_candidates = calib_candidates[:300]  # cap to ~300 distinct images
    with open(calib_list, 'w') as f:
        for p in calib_candidates:
            f.write(str(p).replace('\\', '/') + "\n")

    # Also create a calib_images folder and copy images there for hailomz --calib-path
    calib_dir = work_dir / "calib_images"
    calib_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(calib_candidates):
        dst = calib_dir / f"calib_{i:04d}.jpg"
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print("[warn] failed to copy calib image:", src, "->", dst, ":", e)

    return ds, calib_list, calib_dir

# --------------------------------- Training ----------------------------------

def train_and_export(work_dir: Path, run_name: str, data_yaml: Path,
                     imgsz: int, epochs: int, batch: int, device: str,
                     model_name: str = 'yolov8n.pt'):
    runs_dir = work_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)  # Pi/Hailo-friendly backbone by default

    results = model.train(
        data=str(data_yaml),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        project=str(runs_dir),
        name=run_name,
        device=device,
        # Reasonable defaults
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.3,
        degrees=0.0, translate=0.08, scale=0.5, shear=0.0, flipud=0.0, fliplr=0.5,
        mosaic=0.7, mixup=0.15,
        patience=35,
        save_period=10,
        verbose=True,
    )

    best_pt = runs_dir / run_name / "weights" / "best.pt"
    if not best_pt.exists():
        raise RuntimeError("best.pt not found after training.")

    export_dir = work_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Export ONNX (no NMS, static size) — for Pi app & Hailo
    m = YOLO(str(best_pt))
    onnx_path = m.export(
        format='onnx',
        imgsz=imgsz,
        opset=12,
        dynamic=False,
        simplify=True,
        nms=False
    )
    onnx_path = Path(onnx_path)
    target = export_dir / "best.onnx"
    shutil.copy2(onnx_path, target)

    # Also ONNX with NMS for quick desktop sanity checks
    onnx_nms_path = m.export(
        format='onnx',
        imgsz=imgsz,
        opset=12,
        dynamic=False,
        simplify=True,
        nms=True
    )
    onnx_nms_path = Path(onnx_nms_path)
    target_nms = export_dir / "best_nms.onnx"
    shutil.copy2(onnx_nms_path, target_nms)

    print("\n[OK] Training complete.")
    print(" best.pt        :", best_pt)
    print(" best.onnx      :", target, "(no NMS; use for Pi app & Hailo)")
    print(" best_nms.onnx  :", target_nms, "(with NMS; sanity checks)")

    return best_pt, target, target_nms

# ----------------------------------- Main ------------------------------------

def main():
    # Fallback to CPU if CUDA not available
    dev = DEVICE
    if dev.lower() not in ("cpu",) and not torch.cuda.is_available():
        print("[INFO] CUDA not available; forcing device='cpu'.")
        dev = "cpu"

    print("[CONFIG - HARDCODED WANZL BADGE RUN]")
    print(" POS_DIR :", POS_DIR)
    print(" NEG_DIR :", NEG_DIR)
    print(" WORK_DIR:", WORK_DIR)
    print(" RUN_NAME:", RUN_NAME)
    print(" IMG_SZ  :", IMG_SIZE)
    print(" EPOCHS  :", EPOCHS, " BATCH:", BATCH)
    print(" DEVICE  :", dev)
    print(" MODEL   :", MODEL_NAME)
    print(" POS_FRAC:", POS_FRACTION)

    if not POS_DIR.exists():
        raise FileNotFoundError(f"POS_DIR missing: {POS_DIR}")
    if not NEG_DIR.exists():
        raise FileNotFoundError(f"NEG_DIR missing: {NEG_DIR}")
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Build dataset + calibration set
    ds_root, calib_list, calib_dir = build_dataset(
        POS_DIR, NEG_DIR, WORK_DIR,
        IMG_SIZE, TRAIN_COUNT, VAL_COUNT,
        pos_fraction=POS_FRACTION
    )

    best_pt, best_onnx, best_onnx_nms = train_and_export(
        WORK_DIR, RUN_NAME, ds_root / 'data.yaml',
        IMG_SIZE, EPOCHS, BATCH, dev, MODEL_NAME
    )

    print("\n[HAILO CALIB INFO]")
    print(" calib_list.txt :", calib_list)
    print(" calib_images   :", calib_dir)

    # avoid f-string backslash issue: precompute the normalized string
    exports_dir = WORK_DIR / "exports"
    exports_dir_str = str(exports_dir).replace("\\", "/")


if __name__ == '__main__':
    main()
