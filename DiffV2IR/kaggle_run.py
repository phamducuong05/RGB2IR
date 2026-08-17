# -*- coding: utf-8 -*-
"""
================================================================================
 DIFFV2IR — RUNBOOK CHẠY TRÊN KAGGLE (notebook)
================================================================================
File này là "bản sao" của notebook Kaggle. Mỗi CELL được chia bởi marker `# %%`
(đọc được bởi VSCode / PyCharm / jupytext) + banner `# ===== CELL N =====`.
Các lệnh shell được gói qua hàm `sh()` (subprocess) để file này vừa là .py hợp
lệ, vừa chạy được trong từng cell của notebook Kaggle.

CÁCH DÙNG TRÊN KAGGLE:
  1. Upload dataset FLIR (zip chứa JPEGImages/  seg/  align_validation.txt).
  2. Code DiffV2IR KHÔNG upload zip — dùng GIT CLONE vào /kaggle/working/ (dễ
     cập nhật bằng `git pull`). Yêu cầu trước: commit + push code lên GitHub
     (nhớ push CẢ infer_flir.py). Chi tiết ở CELL 2 + CELL 3.
  3. New Notebook -> GPU T4 x2 -> "+ Add Input" chọn dataset FLIR.
  4. SỬA CELL 2: điền INPUT_FLIR, GIT_REPO_URL, REPO_DIR + bật/tắt wandb.
  5. Chạy lần lượt từng cell:
       CELL 1 : cài đặt
       CELL 2 : khai báo đường dẫn + tham số (SỬA Ở ĐÂY)
       CELL 3 : git clone code DiffV2IR vào /kaggle/working
       CELL 4 : xem cây /kaggle/input + kiểm tra đường dẫn
       CELL 5 : tải trọng số FLIR.ckpt 7.7GB + login wandb (nếu bật)
       CELL 6 : smoke test 5 ảnh (kiểm tra chạy thông, ~3-5 phút)
       CELL 7 : test 20 ảnh  (đánh giá chất lượng: PSNR/SSIM/LPIPS + ảnh so sánh)
       CELL 8 : full validation (~1000 ảnh, có FID)
       CELL 9 : (tùy chọn) chỉ tính lại metrics — không chạy lại diffusion

Trọng số tự tải, không cần upload thủ công:
       DiffV2IR FLIR ckpt: https://huggingface.co/datasets/Lidong26/IR-500K/
                           tree/main/IR-500k/finetuned_checkpoints  -> FLIR.ckpt (7.7 GB)
       BLIP caption     : transformers tự tải từ HF Hub (Salesforce/blip-image-captioning-base).
                          KHÔNG cần file .pth — link GCS gốc của BLIP đã bị Salesforce khóa (403).
       CLIP ViT-L/14    : transformers tự tải từ HuggingFace (openai/clip-vit-large-patch14)

Lưu ý FID:
       - Dưới 50 ảnh (mặc định --fid-min-images) -> FID = nan (đúng thiết kế).
       - Cell 7 chạy 20 ảnh nên FID sẽ là nan — xem PSNR/SSIM/LPIPS là đủ.
       - Cell 8 chạy full (~1000 ảnh) mới có FID đáng tin.

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
#
# LƯU Ý quan trọng — lỗi "Building wheel for tokenizers":
#   Kaggle hiện dùng Python 3.12. Các bản pin cũ (transformers 4.26.1, kornia 0.6,
#   omegaconf 2.1.1, torchmetrics 0.6) ra đời TRƯỚC Python 3.12 nên không có wheel
#   sẵn; tokenizers lại viết bằng Rust -> pip tự biên dịch -> fail.
#   -> Bump lên bản hỗ trợ py3.12, API mà infer_flir.py dùng không đổi.
#   Cũng phải cài thêm `clip` (openai) vì modules.py có `import clip`, và nâng
#   torchmetrics >=0.8 vì code dùng `normalize=True` (torchmetrics 0.6 không có).
#   `taming-transformers` là dependency của stable_diffusion/autoencoder.py
#   (`from taming.modules.vqvae.quantize import VectorQuantizer`) — tác giả cũng
#   ghi trong requirements.txt.
#
#   NẾU CHẠY XONG THẤY "ERROR: pip's dependency resolver does not currently take
#   into account ... dependency conflicts" (kèm danh sách jax, rasterio, shap,
#   sentence-transformers, opencv...) -> ĐÓ LÀ CẢNH BÁO, KHÔNG PHẢI LỖI. Đây là
#   xung đột GIỮA CÁC GÓI KAGGLE ĐÃ CÀI SẴN (numpy 1.26.4 của image vs một số gói
#   muốn numpy>=2.0), không liên quan pipeline của chúng ta. pip vẫn cài xong —
#   cứ chạy tiếp CELL 5.
sh("pip install -q einops omegaconf==2.3.0 torchmetrics==0.11.4 "
   "transformers==4.38.2 kornia==0.7.3 timm lpips "
   "git+https://github.com/crowsonkb/k-diffusion.git "
   "git+https://github.com/openai/CLIP.git@main#egg=clip")
# KHÔNG cần cài taming-transformers: DiffV2IR/taming/ đã được vendor sẵn trong repo
# (chỉ giữ file autoencoder.py cần: quantize.py). Sau git clone là có, không phụ
# thuộc pip cài từ mạng.

# Nâng numpy lên 2.x: image Kaggle cài sẵn jax/opencv/cupy/rasterio... được
# biên dịch CHO numpy 2, nhưng image lại kèm numpy 1.26.4 -> lỗi
# "numpy.dtype size changed" khi import jax. Nâng numpy lên 2.x cho khớp
# (torch trên Kaggle 2026 cũng là bản numpy-2; DiffV2IR dùng numpy rất ít nên an toàn).
sh("pip install -q --upgrade 'numpy>=2.0,<3'")

# %%
# ===================== CELL 2: KHAI BÁO ĐƯỜNG DẪN (SỬA Ở ĐÂY) =====================
# >>> CHỈ CẦN SỬA 4 MỤC: INPUT_FLIR, SEG_DIR, GIT_REPO_URL, REPO_DIR (+ wandb nếu muốn).

# 1. Dataset FLIR chứa JPEGImages/  +  align_validation.txt
#    (RGB ảnh *_RGB.jpg và GT IR *_PreviewData.jpeg nằm TRONG JPEGImages/)
INPUT_FLIR = "/kaggle/input/flir/align"

# 2. Seg map — thường ở dataset RIÊNG (vd phamduccuong05/flir-seg). Điền đúng
#    thư mục chứa các file *_RGB.png (tên trùng ảnh RGB, đuôi .png/.jpg).
SEG_DIR = "/kaggle/input/flir-seg/seg"

# 3. Code DiffV2IR — dùng GIT CLONE vào /kaggle/working/ (không upload zip, dễ cập nhật).
#    a) Trên máy: commit + push code lên GitHub (PHẢI bao gồm infer_flir.py).
#    b) GIT_REPO_URL = URL repo (repo của bạn: phamducuong05/RGB2IR).
#    c) GIT_BRANCH = nhánh chứa code DiffV2IR (đang là v2ir). Để "" = nhánh mặc định.
#    d) REPO_DIR = nơi infer_flir.py nằm SAU khi clone ở CELL 3.
#       Với repo RGB2IR (DiffV2IR là thư mục CON) -> /kaggle/working/RGB2IR/DiffV2IR
#       Nếu repo PRIVATE, kèm token vào URL, vd:
#           GIT_REPO_URL = "https://<username>:<PAT>@github.com/<username>/<repo>.git"
GIT_REPO_URL = "https://github.com/phamducuong05/RGB2IR.git"
GIT_BRANCH   = "v2ir"
REPO_DIR     = "/kaggle/working/RGB2IR/DiffV2IR"

# 3. WANDB (tùy chọn). Muốn log kết quả lên wandb.ai thì:
#      - USE_WANDB = True (bật) / False (tắt hoàn toàn)
#      - Có API key: điền thẳng WANDB_API_KEY bên dưới, HOẶC để trống và khai
#        báo trong Kaggle: Settings (bảng bên phải notebook) -> Secrets ->
#        thêm key tên `WANDB_API_KEY` (Kaggle tự tiêm thành biến môi trường).
#    LƯU Ý: đây là KEY CỦA WANDB (từ https://wandb.ai/authorize), không phải
#    key của Kaggle. Khi BẬT, các cell 6-9 TỰ THÊM cờ --wandb vào lệnh chạy;
#    khi TẮT, lệnh chạy không có --wandb (nhưng vẫn lưu ảnh + metrics bình thường).
USE_WANDB = True
WANDB_API_KEY = ""                       # "" = lấy từ env var WANDB_API_KEY (Kaggle secret)

# ---- Các tham số mặc định (không cần sửa nếu chưa rõ) ----
WEIGHTS_DIR = "/kaggle/working/weights"     # trọng số tải về (cell 5)
CKPT        = WEIGHTS_DIR + "/FLIR.ckpt"
# BLIP caption dùng transformers (tự tải từ HF Hub) — KHÔNG cần file .pth,
# vì link GCS gốc của BLIP đã bị Salesforce khóa (403).
BLIP_MODEL  = "Salesforce/blip-image-captioning-base"

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

# Chỉ khi BẬT wandb (USE_WANDB=True) VÀ có key thì mới gắn cờ --wandb.
# Nếu không, WANDB_ARGS = "" nên các lệnh infer_flir.py chạy không có --wandb.
_EFFECTIVE_KEY = (WANDB_API_KEY or os.environ.get("WANDB_API_KEY", "")).strip()
WANDB_ARGS = (f"--wandb --wandb-project {WANDB_PROJECT}"
              if (USE_WANDB and _EFFECTIVE_KEY) else "")

# %%
# ===================== CELL 3: GIT CLONE CODE VÀO /kaggle/working =====================
# Lấy code DiffV2IR từ GitHub (gồm taming/ đã vendor). Clone vào /kaggle/working/ —
# không phải /kaggle/input/ (chỉ đọc). Có resume:
#   - infer_flir.py + taming/ đều có  -> bỏ qua (chạy lại nhanh).
#   - có infer_flir.py NHƯNG thiếu taming/ -> clone CŨ (bản lỗi taming) -> git pull.
#   - chưa có gì -> clone mới.
os.makedirs("/kaggle/working", exist_ok=True)

TAMING_OK = os.path.isfile(os.path.join(REPO_DIR, "taming", "modules", "vqvae", "quantize.py"))

if os.path.isfile(os.path.join(REPO_DIR, "infer_flir.py")):
    if TAMING_OK:
        print(">> Code đã có tại", REPO_DIR, "— bỏ qua clone.")
    else:
        print(">> Có code nhưng THIẾU taming/ (clone cũ) — git pull để lấy bản đã fix:")
        sh("git -C " + REPO_DIR + " pull")
else:
    os.chdir("/kaggle/working")
    clone_cmd = "git clone " + GIT_REPO_URL + ((" -b " + GIT_BRANCH) if GIT_BRANCH else "")
    sh(clone_cmd)

# Kiểm tra lại lần cuối
TAMING_OK = os.path.isfile(os.path.join(REPO_DIR, "taming", "modules", "vqvae", "quantize.py"))
if os.path.isfile(os.path.join(REPO_DIR, "infer_flir.py")):
    print(">> infer_flir.py:", "OK" if os.path.isfile(os.path.join(REPO_DIR, "infer_flir.py")) else "THIẾU")
    print(">> taming/     :", "OK (vendor)" if TAMING_OK else "THIẾU — kiểm tra GIT_BRANCH/GIT_REPO_URL ở CELL 2")
else:
    print("\n!! KHÔNG tìm thấy infer_flir.py tại", REPO_DIR)
    print("   Chạy `!ls /kaggle/working` để xem tên thư mục sau khi clone,")
    print("   rồi sửa REPO_DIR (và GIT_BRANCH nếu cần) trong CELL 2, và chạy lại cell này.")

# %%
# ===================== CELL 4: XEM CẤU TRÚC /kaggle/input + KIỂM TRA PATH =====================
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
print("SEG_DIR    :", SEG_DIR, "->", "OK" if (os.path.isdir(SEG_DIR) and os.listdir(SEG_DIR)) else "SAI (thiếu/trống seg/)")
print("REPO_DIR   :", REPO_DIR, "->", "OK" if os.path.isfile(os.path.join(REPO_DIR, "infer_flir.py")) else "SAI (thiếu infer_flir.py — chạy CELL 3 trước)")
print("\nNếu có dòng 'SAI' -> sửa lại CELL 2 rồi chạy lại cell này.")

# %%
# ===================== CELL 5: TẢI TRỌNG SỐ (FLIR.ckpt 7.7GB) =====================
# FLIR.ckpt tải trực tiếp từ HuggingFace dataset của tác giả DiffV2IR (gồm luôn
# UNet + VAE encoder/decoder). BLIP caption tự tải khi chạy infer (xem dưới).
# CLIP ViT-L/14 do transformers tự tải khi chạy infer (không tải ở đây).
# ~7.7 GB nên mất vài phút. Mất session thì chạy lại cell này (có check resume).
os.makedirs(WEIGHTS_DIR, exist_ok=True)

if not os.path.isfile(CKPT) or os.path.getsize(CKPT) < 7_000_000_000:
    sh('wget -q -O "' + CKPT + '" '
       '"https://huggingface.co/datasets/Lidong26/IR-500K/resolve/main/'
       'IR-500k/finetuned_checkpoints/FLIR.ckpt"')

# BLIP caption KHÔNG tải ở đây — transformers tự tải từ HF Hub khi chạy infer
# (model: BLIP_MODEL). Không cần file .pth (link GCS gốc của BLIP đã bị khóa).

print("FLIR.ckpt :", f"{os.path.getsize(CKPT)/1e9:.1f} GB" if os.path.isfile(CKPT) else "MISSING")
print("BLIP      :", "tự tải khi chạy infer từ HF Hub ->", BLIP_MODEL)

# Login wandb chỉ khi BẬT (USE_WANDB) và có API key (điền trực tiếp hoặc qua Kaggle secret).
if _EFFECTIVE_KEY:
    os.environ["WANDB_API_KEY"] = _EFFECTIVE_KEY
    import wandb
    wandb.login()   # đọc key từ env var vừa set
    print(">> Wandb: BẬT — cell 6-9 sẽ chạy kèm --wandb (project:", WANDB_PROJECT + ")")
else:
    print(">> Wandb: TẮT — chưa có API key. Các cell 6-9 chạy KHÔNG có --wandb,\n"
          "   nhưng vẫn sinh ảnh + metrics + visualization đầy đủ.")

# %%
# ===================== CELL 6: SMOKE TEST 5 ẢNH =====================
# Chạy thử 5 ảnh để chắc model load + sampling chạy thông (không tính metrics).
# Xem log có dòng "prompt : ..." và 5 ảnh pred xuất hiện trong OUTPUT.
os.chdir(REPO_DIR)                      # để import được stable_diffusion, blip_models
os.makedirs(OUTPUT, exist_ok=True)

sh(f"""python infer_flir.py \
    --input-rgb  {INPUT_FLIR}/JPEGImages \
    --seg-dir    {SEG_DIR} \
    --val-txt    {INPUT_FLIR}/align_validation.txt \
    --gt-dir     {INPUT_FLIR}/JPEGImages \
    --config     {REPO_DIR}/configs/generate.yaml \
    --ckpt       {CKPT} \
    --blip-model {BLIP_MODEL} \
    --output     {OUTPUT} \
    --resolution {RESOLUTION} --steps {STEPS} \
    --cfg-text {CFG_TEXT} --cfg-image {CFG_IMAGE} --cfg-seg {CFG_SEG} \
    --seed {SEED} --vis-num {VIS_NUM} \
    --limit 5 {WANDB_ARGS}""")

# %%
# ===================== CELL 7: TEST 20 ẢNH (ĐÁNH GIÁ NHANH) =====================
# Chạy 20 ảnh -> có PSNR/SSIM/LPIPS + 4 panel so sánh + 20 triplet.
# FID sẽ là nan vì dưới 50 ảnh (mặc định) — đó là đúng thiết kế.
# Xem kết quả: metrics in trong log, ảnh trong {OUTPUT}/visualization/
# (panel_*.png + triplets/) — cũng hiện trên wandb nếu đã bật wandb.
os.chdir(REPO_DIR)

sh(f"""python infer_flir.py \
    --input-rgb  {INPUT_FLIR}/JPEGImages \
    --seg-dir    {SEG_DIR} \
    --val-txt    {INPUT_FLIR}/align_validation.txt \
    --gt-dir     {INPUT_FLIR}/JPEGImages \
    --config     {REPO_DIR}/configs/generate.yaml \
    --ckpt       {CKPT} \
    --blip-model {BLIP_MODEL} \
    --output     {OUTPUT} \
    --resolution {RESOLUTION} --steps {STEPS} \
    --cfg-text {CFG_TEXT} --cfg-image {CFG_IMAGE} --cfg-seg {CFG_SEG} \
    --seed {SEED} --vis-num {VIS_NUM} \
    --limit 20 {WANDB_ARGS}""")

# %%
# ===================== CELL 8: CHẠY FULL VALIDATION =====================
# Chạy toàn bộ align_validation.txt (bỏ --limit) -> ~1000 ảnh, vài giờ.
# FID lúc này mới có nghĩa. Các ảnh đã sinh ở cell 6/7 sẽ được bỏ qua (resume).
os.chdir(REPO_DIR)

sh(f"""python infer_flir.py \
    --input-rgb  {INPUT_FLIR}/JPEGImages \
    --seg-dir    {SEG_DIR} \
    --val-txt    {INPUT_FLIR}/align_validation.txt \
    --gt-dir     {INPUT_FLIR}/JPEGImages \
    --config     {REPO_DIR}/configs/generate.yaml \
    --ckpt       {CKPT} \
    --blip-model {BLIP_MODEL} \
    --output     {OUTPUT} \
    --resolution {RESOLUTION} --steps {STEPS} \
    --cfg-text {CFG_TEXT} --cfg-image {CFG_IMAGE} --cfg-seg {CFG_SEG} \
    --seed {SEED} --vis-num {VIS_NUM} \
    {WANDB_ARGS}""")

# %%
# ===================== CELL 9 (TÙY CHỌN): CHỈ TÍNH LẠI METRICS =====================
# Dùng khi đã có sẵn các ảnh *_pred.png trong OUTPUT — không cần chạy lại diffusion.
# Thích hợp khi bạn muốn thử đổi --fid-min-images hoặc xem lại metrics/visualization.
os.chdir(REPO_DIR)

sh(f"""python infer_flir.py \
    --input-rgb  {INPUT_FLIR}/JPEGImages \
    --seg-dir    {SEG_DIR} \
    --val-txt    {INPUT_FLIR}/align_validation.txt \
    --gt-dir     {INPUT_FLIR}/JPEGImages \
    --config     {REPO_DIR}/configs/generate.yaml \
    --ckpt       {CKPT} \
    --blip-model {BLIP_MODEL} \
    --output     {OUTPUT} \
    --vis-num {VIS_NUM} \
    --metrics-only {WANDB_ARGS}""")
