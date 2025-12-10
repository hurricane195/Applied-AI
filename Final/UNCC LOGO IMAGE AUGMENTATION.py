
# Safe augmentations for *full-word* logos (no partial crops, no mirror flips)

import os
import cv2
from tqdm import tqdm
import albumentations as A
from collections import Counter


SRC_DIR = r"C:...."
DST_DIR = r"C:...."

TARGET_TOTAL = 700
IMG_SIZE = 224
UPSIDE_DOWN_PER_IMAGE = 1   # set to 1–2 ONLY if upside-down logos are truly expected
UD_JITTER_DEG = 5

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic")
DEBUG = False

# ---------- Pipelines ----------
# Keep entire word: scale to fit, pad to square, then light geometric tweaks with fit_output
SAFE_GEOM = [
    A.LongestMaxSize(max_size=int(IMG_SIZE * 0.9)),  # leave margin
    A.PadIfNeeded(IMG_SIZE, IMG_SIZE,
                  border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
    A.Affine(
        translate_percent=(0.0, 0.02),    # tiny drift
        scale=(0.90, 1.10),
        rotate=(-12, 12),
        shear=(-6, 6),
        fit_output=True,
        mode=cv2.BORDER_CONSTANT, cval=(0, 0, 0),
        p=0.85
    ),
    A.Perspective(scale=(0.02, 0.05), fit_output=True, p=0.25),
    A.PadIfNeeded(IMG_SIZE, IMG_SIZE,
                  border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
    A.CenterCrop(IMG_SIZE, IMG_SIZE),
]

PHOTO_JITTER = [
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 3)),
        A.ISONoise(intensity=(0.01, 0.04)),
        A.ImageCompression(quality_lower=70, quality_upper=95),
    ], p=0.5),
    A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.02, p=0.8),
    A.CLAHE(p=0.1),
]

AUGS_BASE = A.Compose(SAFE_GEOM + PHOTO_JITTER, p=1.0)

# Upside-down variant (off by default)
AUGS_UD = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMG_SIZE * 0.9)),
        A.PadIfNeeded(IMG_SIZE, IMG_SIZE,
                      border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
        A.Affine(
            rotate=(180 - UD_JITTER_DEG, 180 + UD_JITTER_DEG),
            fit_output=True,
            mode=cv2.BORDER_CONSTANT, cval=(0, 0, 0),
            p=1.0,
        ),
        A.PadIfNeeded(IMG_SIZE, IMG_SIZE,
                      border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
        A.CenterCrop(IMG_SIZE, IMG_SIZE),
    ] + PHOTO_JITTER,
    p=1.0
)

# ---------- Helpers ----------
def is_image(fn): return os.path.splitext(fn)[1].lower() in EXTS

def collect_images(root):
    items, ext_counter, any_files = [], Counter(), 0
    for dp, _, fns in os.walk(root):
        rel = os.path.relpath(dp, root); rel = "" if rel == "." else rel
        if rel.startswith("."): continue
        for fn in fns:
            any_files += 1
            ext = os.path.splitext(fn)[1].lower(); ext_counter[ext] += 1
            if is_image(fn):
                items.append((os.path.join(dp, fn), rel, os.path.splitext(fn)[0]))
    if DEBUG:
        print(f"[Debug] Files seen: {any_files}")
        for e, c in ext_counter.most_common(): print(f"  {e or '(no ext)'} : {c}")
        if not items: print("[Debug] No images with allowed extensions found.")
    return items

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def save_jpg(path, img):
    return bool(cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95]))

def plan_counts(n, total):
    base = total // n; rem = total % n
    out = [base] * n
    for i in range(rem): out[i] += 1
    return out

# ---------- Main ----------
def main():
    print(f"[Info] Albumentations: {A.__version__}")
    imgs = collect_images(SRC_DIR)
    if not imgs: raise SystemExit(f"No images in: {SRC_DIR}")

    n = len(imgs)
    base_counts = plan_counts(n, TARGET_TOTAL)
    ud_counts = [UPSIDE_DOWN_PER_IMAGE] * n
    ensure_dir(DST_DIR)

    total = 0
    for idx, (abspath, rel_dir, base) in enumerate(tqdm(imgs, desc="Augmenting")):
        img = cv2.imread(abspath)
        if img is None:
            print(f"[Warn] read failed: {abspath}"); continue

        out_dir = os.path.join(DST_DIR, rel_dir); ensure_dir(out_dir)

        # Base (safe) augments
        for i in range(base_counts[idx]):
            aug = AUGS_BASE(image=img)["image"]
            out = os.path.join(out_dir, f"{base}_aug{i:02d}.jpg")
            if save_jpg(out, aug): total += 1

        # Optional upside-down
        for j in range(ud_counts[idx]):
            aug = AUGS_UD(image=img)["image"]
            out = os.path.join(out_dir, f"{base}_ud{j:02d}.jpg")
            if save_jpg(out, aug): total += 1

    print(f"[Done] Wrote {total} images -> {DST_DIR}")

if __name__ == "__main__":
    main()
