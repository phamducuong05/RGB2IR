# -*- coding: utf-8 -*-
"""
SAM segmentation for FLIR — run on Kaggle
==========================================
Creates a NEW dataset: a `seg/` folder with ONE binary segmentation map per
FLIR RGB image, keeping the original RGB filename (e.g. FLIR_00001_RGB.png).

The map is white where SAM detected any object and black everywhere else —
exactly the result the DiffV2IR author gets from SAM + `process_masks.py`, but
masks are merged in memory (no thousands of intermediate PNGs).

New dataset produced (under /kaggle/working):
    seg/                    <- the new dataset
        FLIR_00001_RGB.png  <- seg map for FLIR_00001_RGB.jpg
        FLIR_00002_RGB.png
        ...
    flir_seg.zip            <- the dataset zipped, for download

How to run on Kaggle
--------------------
1. Upload FLIR to Kaggle: Datasets -> New Dataset -> upload a zip of your
   `align` folder (e.g. name it `flir`).
2. Create a Kaggle Notebook with a GPU (Settings -> Accelerator -> GPU T4 x2).
3. "+ Add Input" -> select your `flir` dataset.
4. Paste this whole file into ONE code cell and run. (Or upload it as a
   dataset and run `!python sam_kaggle.py`.)
5. After the run, download flir_seg.zip from the notebook output
   (the "Output" tab / Save Version) -> that is your new seg dataset.

Command-line args (all optional; anything omitted keeps the CONFIG block):
    !python sam_kaggle.py --test-limit 3              # smoke test first
    !python sam_kaggle.py                             # full run (ViT-L default)
    !python sam_kaggle.py --input /kaggle/input/flir/align
    !python sam_kaggle.py --points-per-side 16        # ~2x faster, coarser masks
    !python sam_kaggle.py --filter FLIR_00            # only one name-range (splitting)
    !python sam_kaggle.py --model vit_b               # smaller/faster SAM
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

# ============================== CONFIG ==================================
# Path to your FLIR `align` folder. Leave None to auto-detect under
# /kaggle/input/<dataset>/align .
INPUT_ALIGN = None

# SAM checkpoint (ViT-L, ~1.2 GB). Change these two for a different size.
SAM_MODEL_TYPE = "vit_l"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"

# Where the new dataset goes. Keep /kaggle/working so you can download it.
BASE_OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "./output"

# Name of the seg folder (the new dataset).
SEG_FOLDER = "seg"

# Only keep filenames containing this substring ("" = all). Handy for splitting
# a long run across 2 sessions: session 1 FILENAME_FILTER="FLIR_00",
# session 2 "FLIR_01", then merge the two downloaded seg/ folders.
FILENAME_FILTER = ""

# Process only the first N images (quick smoke test). 0 = all.
TEST_LIMIT = 0

# Skip images whose seg PNG already exists (resume after a kernel restart).
RESUME = True

# Automatic-mask-generator settings (SAM getting-started defaults).
# For the "quality" variant from the SAM notebook (more crops, slower) use:
#   points_per_side=32, pred_iou_thresh=0.88, stability_score_thresh=0.95,
#   crop_n_layers=1, crop_n_points_downscale_factor=2, min_mask_region_area=100
# For ~2x speed on ViT-L try points_per_side=16.
MASK_GEN_PARAMS = dict(
    points_per_side=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
    crop_n_layers=0,
    crop_n_points_downscale_factor=1,
    min_mask_region_area=100,
)
# ========================================================================

SEG_DIR = os.path.join(BASE_OUT, SEG_FOLDER)


def ensure_deps():
    """Install segment-anything if missing; verify opencv (SAM needs it)."""
    try:
        import segment_anything  # noqa: F401
    except ImportError:
        print(">> Installing segment-anything ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q",
             "git+https://github.com/facebookresearch/segment-anything.git"]
        )
    import cv2  # noqa: F401  (used by SAM for min_mask_region_area)
    import torch  # noqa: F401
    from tqdm.auto import tqdm
    return tqdm


def download_checkpoint():
    ckpt = os.path.join(BASE_OUT, os.path.basename(CHECKPOINT_URL))
    if os.path.isfile(ckpt) and os.path.getsize(ckpt) > 1_000_000_000:
        return ckpt
    print(f">> Downloading SAM checkpoint ({os.path.basename(CHECKPOINT_URL)}) ...")
    os.makedirs(BASE_OUT, exist_ok=True)
    subprocess.check_call(["wget", "-q", "-O", ckpt, CHECKPOINT_URL])
    return ckpt


def find_align_root():
    """Locate the FLIR align folder containing JPEGImages."""
    if INPUT_ALIGN:
        if not os.path.isdir(os.path.join(INPUT_ALIGN, "JPEGImages")):
            raise SystemExit(f"INPUT_ALIGN has no JPEGImages subfolder: {INPUT_ALIGN}")
        return INPUT_ALIGN
    for cand in sorted(glob.glob("/kaggle/input/*/align")) + sorted(glob.glob("/kaggle/input/*")):
        if os.path.isdir(os.path.join(cand, "JPEGImages")):
            return cand
    raise SystemExit(
        "Cannot find FLIR align folder. Set INPUT_ALIGN to the path of your "
        "align folder (it must contain JPEGImages/)."
    )


def discover_rgb_images(align_root):
    """Return the list of RGB (visible) image paths in the align folder."""
    jpeg_dir = os.path.join(align_root, "JPEGImages")
    rgb = []
    for pat in ("*_RGB.jpg", "*_RGB.jpeg", "*_RGB.png", "*_RGB.JPG"):
        rgb += glob.glob(os.path.join(jpeg_dir, pat))
    rgb = sorted(set(rgb))

    if not rgb:  # fallback: some FLIR layouts keep RGB in align/RGB
        rgb_dir = os.path.join(align_root, "RGB")
        if os.path.isdir(rgb_dir):
            rgb = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")) +
                         glob.glob(os.path.join(rgb_dir, "*.jpeg")) +
                         glob.glob(os.path.join(rgb_dir, "*.png")))

    if not rgb:
        print("Contents of JPEGImages (first 10):",
              os.listdir(jpeg_dir)[:10] if os.path.isdir(jpeg_dir) else "no JPEGImages dir")
        raise SystemExit(
            "No RGB images found. Check the *_RGB.jpg naming inside align/JPEGImages "
            "and adjust discover_rgb_images()."
        )
    if FILENAME_FILTER:
        rgb = [p for p in rgb if FILENAME_FILTER in os.path.basename(p)]
    return rgb


CHECKPOINTS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}


def parse_args():
    """Command-line overrides for the CONFIG block. Any omitted arg keeps the
    value hard-coded at the top of this file."""
    p = argparse.ArgumentParser(description="Generate SAM seg maps for FLIR on Kaggle")
    p.add_argument("--input", help="Path to FLIR align folder (overrides INPUT_ALIGN)")
    p.add_argument("--filter", help="Only process filenames containing this substring")
    p.add_argument("--test-limit", type=int, default=0,
                   help="Process only the first N images as a smoke test (0 = all)")
    p.add_argument("--no-resume", action="store_true",
                   help="Re-generate seg maps even if they already exist")
    p.add_argument("--model", choices=list(CHECKPOINTS), default=SAM_MODEL_TYPE,
                   help="SAM model size (default: %(default)s)")
    p.add_argument("--points-per-side", type=int,
                   help="SAM prompt grid per side (16 = ~2x faster, 32 = default quality)")
    p.add_argument("--out", help="Output directory for the new dataset")
    return p.parse_args()


def main():
    global INPUT_ALIGN, FILENAME_FILTER, TEST_LIMIT, RESUME, SEG_DIR, BASE_OUT, \
        SAM_MODEL_TYPE, CHECKPOINT_URL, MASK_GEN_PARAMS
    args = parse_args()
    if args.input:
        INPUT_ALIGN = args.input
    if args.filter is not None:
        FILENAME_FILTER = args.filter
    if args.test_limit:
        TEST_LIMIT = args.test_limit
    if args.no_resume:
        RESUME = False
    if args.model != SAM_MODEL_TYPE:
        SAM_MODEL_TYPE = args.model
        CHECKPOINT_URL = CHECKPOINTS[args.model]
    if args.out:
        BASE_OUT = args.out
        SEG_DIR = os.path.join(BASE_OUT, SEG_FOLDER)
    if args.points_per_side:
        MASK_GEN_PARAMS = dict(MASK_GEN_PARAMS, points_per_side=args.points_per_side)

    tqdm = ensure_deps()
    ckpt = download_checkpoint()

    align_root = find_align_root()
    rgb_images = discover_rgb_images(align_root)
    print(f">> align root : {align_root}")
    print(f">> RGB images : {len(rgb_images)}  (filter={FILENAME_FILTER or 'all'})")

    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    import torch

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=ckpt)
    sam.to("cuda")
    sam.eval()
    mask_generator = SamAutomaticMaskGenerator(sam, **MASK_GEN_PARAMS)

    os.makedirs(SEG_DIR, exist_ok=True)
    manifest, failures = [], []
    done, skipped = 0, 0
    start = time.time()

    for rgb_path in tqdm(rgb_images, desc="SAM masking", unit="img"):
        stem = os.path.splitext(os.path.basename(rgb_path))[0]
        seg_path = os.path.join(SEG_DIR, stem + ".png")
        entry = [os.path.basename(rgb_path), stem + ".png"]

        if RESUME and os.path.isfile(seg_path) and os.path.getsize(seg_path) > 0:
            manifest.append(entry)
            skipped += 1
            continue
        if TEST_LIMIT and done >= TEST_LIMIT:
            break

        try:
            image = np.array(Image.open(rgb_path).convert("RGB"))
            with torch.inference_mode():
                masks = mask_generator.generate(image)
            merged = np.zeros(image.shape[:2], bool)  # union of all masks
            for m in masks:
                merged |= m["segmentation"]
            seg_img = np.zeros((*image.shape[:2], 3), np.uint8)
            seg_img[merged] = 255  # white object, black background
            Image.fromarray(seg_img).save(seg_path)
            manifest.append(entry)
            done += 1
        except Exception as e:  # keep going: one bad image must not kill the run
            failures.append([os.path.basename(rgb_path), str(e)])
            print(f"!! {os.path.basename(rgb_path)}: {e}")

        if done and (done in (1, 5, 20) or done % 200 == 0):
            elapsed = time.time() - start
            remaining = len(rgb_images) - skipped - done
            print(f"   ... {done} done, ~{elapsed / done:.2f}s/img, "
                  f"ETA ~{elapsed / done * remaining / 60:.1f} min")

    elapsed = time.time() - start
    print(f"\n>> Finished: processed={done}, skipped={skipped}, failed={len(failures)} "
          f"in {elapsed / 60:.1f} min ({elapsed / max(done, 1):.2f}s/img).")

    with open(os.path.join(BASE_OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    with open(os.path.join(BASE_OUT, "failures.json"), "w") as f:
        json.dump(failures, f, indent=1)

    # Zip the seg folder -> this is your downloadable new dataset.
    zip_path = shutil.make_archive(os.path.join(BASE_OUT, "flir_seg"), "zip",
                                   root_dir=BASE_OUT, base_dir=SEG_FOLDER)
    print(f">> New dataset : {SEG_DIR}  ({len(manifest)} seg maps)")
    print(f">> Zip         : {zip_path} ({os.path.getsize(zip_path) / 1e6:.1f} MB)")

    # Quick sanity check on the first seg maps.Hiê
    samples = sorted(glob.glob(os.path.join(SEG_DIR, "*.png")))[:5]
    for s in samples:
        a = np.array(Image.open(s))
        print(f"   {os.path.basename(s)}  size={a.shape[:2]}  "
              f"white_ratio={np.mean(a > 0):.3f}")
    if failures:
        print(f"\n>> {len(failures)} failures in failures.json — first: {failures[0]}")


if __name__ == "__main__":
    main()
