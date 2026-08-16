# -*- coding: utf-8 -*-
"""
================================================================================
 DIFFV2IR — RUNBOOK CHẠY TRÊN KAGGLE (notebook)
================================================================================
File này là "bản sao" của notebook Kaggle. Mỗi CELL được chia bởi marker `# %%`
(đọc được bởi VSCode / PyCharm / jupytext) + banner `# ===== CELL N =====`.
Các lệnh shell được gói qua hàm `sh()` (subprocess) để file này vừa là .py hợp
lệ, vừa chạy được trong từng cell của notebook Kaggle.

CÁCH DÙNG TRÊN KAGGLE (chỉ cần upload data + sửa path ở CELL 2):
  1. Upload 2 thứ lên Kaggle (Datasets -> New Dataset):
       - Dataset FLIR : chứa JPEGImages/  seg/  align_validation.txt
       - Dataset code  : thư mục DiffV2IR (infer_flir.py, stable_diffusion/,
                        blip_models/, configs/)
  2. New Notebook -> GPU T4 x2 -> "+ Add Input" chọn 2 dataset trên.
  3. SỬA CELL 2: khai báo đúng 2 đường dẫn đầu (tên dataset bạn đặt).
  4. Chạy lần lượt từng cell:
       CELL 1 : cài đặt
       CELL 2 : khai báo đường dẫn + tham số (SỬA Ở ĐÂY)
       CELL 3 : xem cây /kaggle/input (đối chiếu path đã khai)
       CELL 4 : tải trọng số (FLIR.ckpt 7.7GB + BLIP) + login wandb (nếu có key)
       CELL 5 : smoke test 5 ảnh (kiểm tra chạy thông, ~3-5 phút)
       CELL 6 : test 20 ảnh  (đánh giá chất lượng: PSNR/SSIM/LPIPS + ảnh so sánh)
       CELL 7 : full validation (~1000 ảnh, có FID)
       CELL 8 : (tùy chọn) chỉ tính lại metrics — không chạy lại diffusion

Trọng số tự tải, không cần upload thủ công:
       DiffV2IR FLIR ckpt: https://huggingface.co/datasets/Lidong26/IR-500K/
                           tree/main/IR-500k/finetuned_checkpoints  -> FLIR.ckpt (7.7 GB)
       BLIP caption     : https://storage.googleapis.com/sfr-vision-language-research/
                          BLIP/models/model_base_caption.pth
       CLIP ViT-L/14    : transformers tự tải từ HuggingFace (openai/clip-vit-large-patch14)

Lưu ý FID:
       - Dưới 50 ảnh (mặc định --fid-min-images) -> FID = nan (đúng thiết kế).
       - Cell 6 chạy 20 ảnh nên FID sẽ là nan — xem PSNR/SSIM/LPIPS là đủ.
       - Cell 7 chạy full (~1000 ảnh) mới có FID đáng tin.

Muốn chuyển file này thành notebook: trong VSCode bấm "Run Cell" từng block,
hoặc dùng jupytext:
    pip install jupytext && jupytext --to ipynb kaggle_run.py
================================================================================
"""

import os
import subprocess
import sys


def sh(cmd, check=True):
    """Chạy lệnh shell và in ra — thay cho `!cmd` trong notebook."""
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=check)


# %%
# ============================= CELL 1: CÀI ĐẶT =============================
# Chạy các lệnh pip trên Kaggle (Chỉ chạy 1 lần; mỗi session Kaggle tách riêng).
sh("pip install -q einops==0.3.0 omegaconf==2.1.1 torchmetrics==0.6.0 "
   "transformers==4.26.1 kornia==0.6 timm "
   "git+https://github.com/crowsonkb/k-diffusion.git")

# %%
# ===================== CELL 2: KHAI BÁO ĐƯỜNG DẪN (SỬA Ở ĐÂY) =====================
# >>> CHỈ CẦN SỬA 3 DÒNG ĐẦU: tên dataset bạn đặt trên Kaggle.
# Kaggle mount dataset vào /kaggle/input/<tên-dataset>.
# Chạy CELL 3 để xem cây /kaggle/input rồi quay lại chỉnh nếu cần.

# 1. Dataset FLIR: thư mục chứa JPEGImages/  seg/  align_validation.txt
INPUT_FLIR = "/kaggle/input/flir/align"

# 2. Dataset code: thư mục chứa infer_flir.py (DiffV2IR repo)
REPO_DIR   = "/kaggle/input/diffv2ir"

# 3. (Tùy chọn) key wandb. Để trống = KHÔNG dùng wandb, chạy được ngay.
#    Điền key -> cell 6/7 sẽ tự log metrics + ảnh so sánh lên wandb.
WANDB_API_KEY = ""

# ---- Các tham số mặc định (không cần sửa nếu chưa rõ) ----
WEIGHTS_DIR = "/kaggle/working/weights"     # trọng số tải về (cell 4)
CKPT        = WEIGHTS_DIR + "/FLIR.ckpt"
BLIP_CKPT   = WEIGHTS_DIR + "/model_base_caption.pth"

RESOLUTION = 512    # cạnh dài ảnh (bắt buộc bội của 64)
STEPS      = 100    # số bước sampling
CFG_TEXT   = 7.5    # CFG text
CFG_IMAGE  = 1.5    # CFG RGB
CFG_SEG    = 1.5    # CFG seg
SEED       = 0
VIS_NUM    = 20     # số ảnh cho ảnh so sánh trực quan
FID_MIN    = 50     # số ảnh tối thiểu để FID có nghĩa (dưới -> nan)
OUTPUT     = "/kaggle/working/out"           # nơi lưu ảnh pred + metrics + visualization
WANDB_PROJECT = "diffv2ir-flir"

# Nếu có key wandb thì các cell 6/7/8 thêm cờ --wandb; ngược lại bỏ qua.
WANDB_ARGS = f"--wandb --wandb-project {WANDB_PROJECT}" if WANDB_API_KEY else ""

# %%
# ===================== CELL 3: XEM CẤU TRÚC /kaggle/input =====================
# Chạy cell này để chắc tên dataset/đường dẫn đúng rồi quay lại sửa Cell 2 nếu cần.
def tree(path, indent=0, depth=2):
    if not os.path.isdir(path):
        print("  " * indent + f"(missing) {path}")
        return
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if os.path.isdir(full):
            print("  " * indent + f"{name}/")
            if depth > 0:
                tree(full, indent + 1, depth - 1)
        else:
            print("  " * indent + f"{name}")

tree("/kaggle/input")
print()
print("INPUT_FLIR :", INPUT_FLIR, "->", "OK" if os.path.isdir(os.path.join(INPUT_FLIR, "JPEGImages")) else "SAI (thiếu JPEGImages/)")
print("REPO_DIR   :", REPO_DIR, "->", "OK" if os.path.isfile(os.path.join(REPO_DIR, "infer_flir.py")) else "SAI (thiếu infer_flir.py)")
print("\nNếu có dòng 'SAI' -> sửa lại 2 dòng đầu của CELL 2 rồi chạy lại cell này.")

# %%
# ===================== CELL 4: TẢI TRỌNG SỐ (FLIR.ckpt 7.7GB + BLIP) =====================
# FLIR.ckpt tải trực tiếp từ HuggingFace dataset của tác giả DiffV2IR (gồm luôn
# UNet + VAE encoder/decoder). BLIP caption tải từ Salesforce.
# CLIP ViT-L/14 do transformers tự tải khi chạy infer (không tải ở đây).
# ~7.7 GB nên mất vài phút. Mất session thì chạy lại cell này (có check resume).
os.makedirs(WEIGHTS_DIR, exist_ok=True)

if not os.path.isfile(CKPT) or os.path.getsize(CKPT) < 7_000_000_000:
    sh('wget -q -O "' + CKPT + '" '
       '"https://huggingface.co/datasets/Lidong26/IR-500K/resolve/main/'
       'IR-500k/finetuned_checkpoints/FLIR.ckpt"')

if not os.path.isfile(BLIP_CKPT):
    sh('wget -q -O "' + BLIP_CKPT + '" '
       '"https://storage.googleapis.com/sfr-vision-language-research/BLIP/'
       'models/model_base_caption.pth"')

print("FLIR.ckpt :", f"{os.path.getsize(CKPT)/1e9:.1f} GB" if os.path.isfile(CKPT) else "MISSING")
print("BLIP      :", f"{os.path.getsize(BLIP_CKPT)/1e9:.1f} GB" if os.path.isfile(BLIP_CKPT) else "MISSING")

# Login wandb chỉ khi bạn đã điền WANDB_API_KEY ở CELL 2.
if WANDB_API_KEY:
    sh("wandb login --relogin " + WANDB_API_KEY)

# %%
# ===================== CELL 5: SMOKE TEST 5 ẢNH =====================
# Chạy thử 5 ảnh để chắc model load + sampling chạy thông (không tính metrics).
# Xem log có dòng "prompt : ..." và 5 ảnh pred xuất hiện trong OUTPUT.
os.chdir(REPO_DIR)                      # để import được stable_diffusion, blip_models
os.makedirs(OUTPUT, exist_ok=True)

sh(f"""python infer_flir.py \
    --input-rgb  {INPUT_FLIR}/JPEGImages \
    --seg-dir    {INPUT_FLIR}/seg \
    --val-txt    {INPUT_FLIR}/align_validation.txt \
    --gt-dir     {INPUT_FLIR}/JPEGImages \
    --config     {REPO_DIR}/configs/generate.yaml \
    --ckpt       {CKPT} \
    --blip-ckpt  {BLIP_CKPT} \
    --output     {OUTPUT} \
    --resolution {RESOLUTION} --steps {STEPS} \
    --cfg-text {CFG_TEXT} --cfg-image {CFG_IMAGE} --cfg-seg {CFG_SEG} \
    --seed {SEED} --vis-num {VIS_NUM} \
    --limit 5 {WANDB_ARGS}""")

# %%
# ===================== CELL 6: TEST 20 ẢNH (ĐÁNH GIÁ NHANH) =====================
# Chạy 20 ảnh -> có PSNR/SSIM/LPIPS + 4 panel so sánh + 20 triplet.
# FID sẽ là nan vì dưới 50 ảnh (mặc định) — đó là đúng thiết kế.
# Xem kết quả: metrics in trong log, ảnh trong {OUTPUT}/visualization/
# (panel_*.png + triplets/) — cũng hiện trên wandb nếu đã điền WANDB_API_KEY.
os.chdir(REPO_DIR)

sh(f"""python infer_flir.py \
    --input-rgb  {INPUT_FLIR}/JPEGImages \
    --seg-dir    {INPUT_FLIR}/seg \
    --val-txt    {INPUT_FLIR}/align_validation.txt \
    --gt-dir     {INPUT_FLIR}/JPEGImages \
    --config     {REPO_DIR}/configs/generate.yaml \
    --ckpt       {CKPT} \
    --blip-ckpt  {BLIP_CKPT} \
    --output     {OUTPUT} \
    --resolution {RESOLUTION} --steps {STEPS} \
    --cfg-text {CFG_TEXT} --cfg-image {CFG_IMAGE} --cfg-seg {CFG_SEG} \
    --seed {SEED} --vis-num {VIS_NUM} \
    --limit 20 {WANDB_ARGS}""")

# %%
# ===================== CELL 7: CHẠY FULL VALIDATION =====================
# Chạy toàn bộ align_validation.txt (bỏ --limit) -> ~1000 ảnh, vài giờ.
# FID lúc này mới có nghĩa. Các ảnh đã sinh ở cell 5/6 sẽ được bỏ qua (resume).
os.chdir(REPO_DIR)

sh(f"""python infer_flir.py \
    --input-rgb  {INPUT_FLIR}/JPEGImages \
    --seg-dir    {INPUT_FLIR}/seg \
    --val-txt    {INPUT_FLIR}/align_validation.txt \
    --gt-dir     {INPUT_FLIR}/JPEGImages \
    --config     {REPO_DIR}/configs/generate.yaml \
    --ckpt       {CKPT} \
    --blip-ckpt  {BLIP_CKPT} \
    --output     {OUTPUT} \
    --resolution {RESOLUTION} --steps {STEPS} \
    --cfg-text {CFG_TEXT} --cfg-image {CFG_IMAGE} --cfg-seg {CFG_SEG} \
    --seed {SEED} --vis-num {VIS_NUM} \
    {WANDB_ARGS}""")

# %%
# ===================== CELL 8 (TÙY CHỌN): CHỈ TÍNH LẠI METRICS =====================
# Dùng khi đã có sẵn các ảnh *_pred.png trong OUTPUT — không cần chạy lại diffusion.
# Thích hợp khi bạn muốn thử đổi --fid-min-images hoặc xem lại metrics/visualization.
os.chdir(REPO_DIR)

sh(f"""python infer_flir.py \
    --input-rgb  {INPUT_FLIR}/JPEGImages \
    --seg-dir    {INPUT_FLIR}/seg \
    --val-txt    {INPUT_FLIR}/align_validation.txt \
    --gt-dir     {INPUT_FLIR}/JPEGImages \
    --config     {REPO_DIR}/configs/generate.yaml \
    --ckpt       {CKPT} \
    --blip-ckpt  {BLIP_CKPT} \
    --output     {OUTPUT} \
    --vis-num {VIS_NUM} \
    --metrics-only {WANDB_ARGS}""")
