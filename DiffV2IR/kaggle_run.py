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
  1. Upload dataset FLIR có thư mục JPEGImages/ chứa toàn bộ ảnh RGB (và nếu có
     GT thì các file *_PreviewData); upload seg map ở Dataset riêng nếu cần.
  2. Code DiffV2IR KHÔNG upload zip — dùng GIT CLONE vào /kaggle/working/ (dễ
     cập nhật bằng `git pull`). Yêu cầu trước: commit + push code lên GitHub
     (nhớ push CẢ infer_flir.py). Chi tiết ở CELL 2 + CELL 3.
  3. New Notebook -> GPU T4 x2 -> "+ Add Input" chọn dataset FLIR, seg và
     `diffv2ir-model-weights-v1` (nếu muốn dùng weight đã đóng gói).
  4. SỬA CELL 2: điền INPUT_FLIR, SEG_DIR, WEIGHTS_DATASET_DIR, GIT_REPO_URL,
     REPO_DIR, khoảng ảnh START_INDEX/END_INDEX, Kaggle secret names và bật/tắt wandb.
  5. Chạy lần lượt từng cell:
       CELL 1 : cài đặt
       CELL 2 : khai báo đường dẫn + tham số (SỬA Ở ĐÂY)
       CELL 3 : git clone code DiffV2IR vào /kaggle/working
       CELL 4 : xem cây /kaggle/input + kiểm tra đường dẫn
       CELL 5 : dùng weight Dataset hoặc tải FLIR.ckpt + login wandb (nếu bật)
       CELL 5B: quét toàn bộ ảnh và chuẩn bị thư mục part
       CELL 6 : chạy toàn bộ ảnh trong [START_INDEX, END_INDEX)
       CELL 7 : kiểm tra part + tạo manifest/metadata
       CELL 8 : upload Dataset private

Nếu không attach Dataset weight, trọng số tự tải, không cần upload thủ công:
       DiffV2IR FLIR ckpt: https://huggingface.co/datasets/Lidong26/IR-500K/
                           tree/main/IR-500k/finetuned_checkpoints  -> FLIR.ckpt (7.7 GB)
       BLIP caption     : transformers tự tải từ HF Hub (Salesforce/blip-image-captioning-base).
                          KHÔNG cần file .pth — link GCS gốc của BLIP đã bị Salesforce khóa (403).
       CLIP ViT-L/14    : transformers tự tải từ HuggingFace (openai/clip-vit-large-patch14)

Muốn chuyển file này thành notebook: trong VSCode bấm "Run Cell" từng block,
hoặc dùng jupytext:
    pip install jupytext && jupytext --to ipynb kaggle_run.py
================================================================================
"""

import os
import json
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
# >>> SỬA INPUT_FLIR, SEG_DIR, khoảng ảnh và thông tin Kaggle; các mục còn lại giữ nguyên.

# 1. Dataset FLIR chứa JPEGImages/ với toàn bộ ảnh RGB và (nếu có) GT IR.
#    Cell 5B sẽ tự quét tất cả ảnh, không cần align_validation.txt.
INPUT_FLIR = "/kaggle/input/flir/align"

# 2. Seg map — thường ở dataset RIÊNG (vd phamduccuong05/flir-seg). Điền đúng
#    thư mục chứa các file *_RGB.png (tên trùng ảnh RGB, đuôi .png/.jpg).
SEG_DIR = "/kaggle/input/flir-seg/seg"

# 3. Khoảng ảnh cần chạy, theo quy ước [START_INDEX, END_INDEX):
#    START được lấy, END không được lấy. Với 5.142 ảnh có thể dùng:
#      [0, 1714), [1714, 3428), [3428, 5142)
START_INDEX = 0
END_INDEX   = 1714

# 4. Kaggle Dataset private sẽ có slug: PREFIX-START-END.
DATASET_SLUG_PREFIX = "diffv2ir-generated"
# Tên hai Kaggle Secrets (giá trị secret là username và API key tương ứng).
KAGGLE_USERNAME_SECRET = "phamduccuong05"
KAGGLE_KEY_SECRET      = "KEY"

# 5. Dataset weights (không bắt buộc). Nếu attach Dataset này, notebook dùng
#    FLIR.ckpt + BLIP + CLIP local và không tải lại từ Hugging Face.
#    Nếu slug khác, sửa đường dẫn; để "" để quay về cơ chế tự tải cũ.
WEIGHTS_DATASET_DIR = "/kaggle/input/diffv2ir-model-weights-v1"

# 6. Code DiffV2IR — dùng GIT CLONE vào /kaggle/working/ (không upload zip, dễ cập nhật).
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

# 7. WANDB (tùy chọn). Muốn log kết quả lên wandb.ai thì:
#      - USE_WANDB = True (bật) / False (tắt hoàn toàn)
#      - Có API key: điền thẳng WANDB_API_KEY bên dưới, HOẶC để trống và khai
#        báo trong Kaggle: Settings (bảng bên phải notebook) -> Secrets ->
#        thêm key tên `WANDB_API_KEY` (Kaggle tự tiêm thành biến môi trường).
#    LƯU Ý: đây là KEY CỦA WANDB (từ https://wandb.ai/authorize), không phải
#    key của Kaggle. Khi BẬT, Cell 6 TỰ THÊM cờ --wandb vào lệnh chạy;
#    khi TẮT, lệnh chạy không có --wandb (nhưng vẫn lưu ảnh + metrics bình thường).
USE_WANDB = True
WANDB_API_KEY = ""                       # "" = lấy từ env var WANDB_API_KEY (Kaggle secret)

# ---- Các tham số mặc định (không cần sửa nếu chưa rõ) ----
WEIGHTS_DIR = "/kaggle/working/weights"     # trọng số tải về (cell 5)
CKPT        = WEIGHTS_DIR + "/FLIR.ckpt"
# BLIP caption dùng transformers — local path nếu có Dataset weights, nếu không
# sẽ tự tải từ HF Hub.
BLIP_MODEL  = "Salesforce/blip-image-captioning-base"
CLIP_MODEL  = "openai/clip-vit-large-patch14"
WEIGHTS_SOURCE = "huggingface"

if WEIGHTS_DATASET_DIR and os.path.isdir(WEIGHTS_DATASET_DIR):
    _dataset_weight_paths = {
        "checkpoint": os.path.join(WEIGHTS_DATASET_DIR, "FLIR.ckpt"),
        "blip": os.path.join(WEIGHTS_DATASET_DIR, "blip-image-captioning-base"),
        "clip": os.path.join(WEIGHTS_DATASET_DIR, "clip-vit-large-patch14"),
    }
    _missing_weight_paths = [
        path for path in _dataset_weight_paths.values()
        if not (os.path.isfile(path) or os.path.isdir(path))
    ]
    if _missing_weight_paths:
        raise FileNotFoundError(
            "Dataset weights thiếu file/thư mục: "
            + ", ".join(_missing_weight_paths)
        )
    if os.path.getsize(_dataset_weight_paths["checkpoint"]) < 7_000_000_000:
        raise FileNotFoundError(
            "FLIR.ckpt trong Dataset weights có vẻ chưa tải đủ 7GB."
        )

    CKPT = _dataset_weight_paths["checkpoint"]
    BLIP_MODEL = _dataset_weight_paths["blip"]
    CLIP_MODEL = _dataset_weight_paths["clip"]
    WEIGHTS_SOURCE = "attached_dataset"

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
print("WEIGHTS    :", WEIGHTS_DATASET_DIR, "->", "OK (attached)" if WEIGHTS_SOURCE == "attached_dataset" else "không attach (Cell 5 sẽ fallback)")
print("\nNếu có dòng 'SAI' -> sửa lại CELL 2 rồi chạy lại cell này.")

# %%
# ===================== CELL 5: CHUẨN BỊ TRỌNG SỐ =====================
# Nếu đã attach Dataset weights, dùng trực tiếp các file read-only trong /kaggle/input.
# Nếu chưa attach, chỉ tải FLIR.ckpt vào /kaggle/working/weights; BLIP/CLIP sẽ tự tải
# từ Hugging Face khi infer_flir.py khởi tạo model.
os.makedirs(WEIGHTS_DIR, exist_ok=True)

if WEIGHTS_SOURCE == "huggingface":
    if not os.path.isfile(CKPT) or os.path.getsize(CKPT) < 7_000_000_000:
        sh('wget -q -O "' + CKPT + '" '
           '"https://huggingface.co/datasets/Lidong26/IR-500K/resolve/main/'
           'IR-500k/finetuned_checkpoints/FLIR.ckpt"')
    print(">> Weight source: Hugging Face / Kaggle working")
else:
    print(">> Weight source: attached Dataset ->", WEIGHTS_DATASET_DIR)

print("FLIR.ckpt :", f"{os.path.getsize(CKPT)/1e9:.1f} GB" if os.path.isfile(CKPT) else "MISSING")
print("BLIP      :", BLIP_MODEL)
print("CLIP      :", CLIP_MODEL)

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
# ===================== CELL 5B: QUÉT TOÀN BỘ ẢNH VÀ CHUẨN BỊ PART =====================
# Dùng khoảng [START_INDEX, END_INDEX) trên toàn bộ ảnh phát hiện trong
# INPUT_FLIR/JPEGImages; không cần align_validation.txt.
sys.path.insert(0, REPO_DIR)
from kaggle_shards import discover_image_keys, select_key_range

ALL_IMAGE_DIR = os.path.join(INPUT_FLIR, "JPEGImages")
ALL_KEYS = discover_image_keys(ALL_IMAGE_DIR)
if not ALL_KEYS:
    raise RuntimeError(f"Không tìm thấy ảnh RGB trong {ALL_IMAGE_DIR}")

ALL_KEYS_TXT = "/kaggle/working/all_image_keys.txt"
with open(ALL_KEYS_TXT, "w", encoding="utf-8") as _f:
    _f.write("\n".join(ALL_KEYS) + "\n")

SELECTED_KEYS = select_key_range(ALL_KEYS, START_INDEX, END_INDEX)
RANGE_TAG = f"{START_INDEX:05d}-{END_INDEX:05d}"
PACKAGE_DIR = f"/kaggle/working/diffv2ir-part-{RANGE_TAG}"
PART_OUTPUT = os.path.join(PACKAGE_DIR, "predictions")
PART_VAL_TXT = f"/kaggle/working/validation_{RANGE_TAG}.txt"

os.makedirs(PART_OUTPUT, exist_ok=True)
with open(PART_VAL_TXT, "w", encoding="utf-8") as _f:
    _f.write("\n".join(SELECTED_KEYS) + "\n")

# Từ đây mọi cell inference dùng đúng part hiện tại.
OUTPUT = PART_OUTPUT
print(f">> Discovered toàn bộ ảnh: {len(ALL_KEYS)}")
print(f">> Range: [{START_INDEX}, {END_INDEX}) -> {len(SELECTED_KEYS)} ảnh")
print(f">> Key đầu: {SELECTED_KEYS[0]} | key cuối: {SELECTED_KEYS[-1]}")
print(">> Selected key list:", PART_VAL_TXT)
print(">> Part output      :", PART_OUTPUT)

# %%
# ===================== CELL 6: CHẠY TOÀN BỘ KHOẢNG ĐÃ CHỌN =====================
# Chạy đúng [START_INDEX, END_INDEX), không chạy smoke/test riêng và không chạy
# toàn bộ validation ngoài khoảng đã chọn. Mỗi notebook chỉ cần đổi START_INDEX,
# END_INDEX ở Cell 2 rồi chạy lại từ Cell 5B.
os.chdir(REPO_DIR)                      # để import được stable_diffusion, blip_models
os.makedirs(OUTPUT, exist_ok=True)
print(f">> Inference range {RANGE_TAG}: {len(SELECTED_KEYS)} ảnh -> {OUTPUT}")

sh(f"""python infer_flir.py \
    --input-rgb  {INPUT_FLIR}/JPEGImages \
    --seg-dir    {SEG_DIR} \
    --val-txt    {PART_VAL_TXT} \
    --gt-dir     {INPUT_FLIR}/JPEGImages \
    --config     {REPO_DIR}/configs/generate.yaml \
    --ckpt       {CKPT} \
    --blip-model {BLIP_MODEL} \
    --clip-version {CLIP_MODEL} \
    --output     {OUTPUT} \
    --resolution {RESOLUTION} --steps {STEPS} \
    --cfg-text {CFG_TEXT} --cfg-image {CFG_IMAGE} --cfg-seg {CFG_SEG} \
    --seed {SEED} --vis-num {VIS_NUM} \
    {WANDB_ARGS}""")

# %%
# ===================== CELL 7: KIỂM TRA PART VÀ ĐÓNG GÓI =====================
# Chỉ chạy cell này SAU CELL 6 (đã chạy xong khoảng đã chọn).
# Nếu thiếu/thừa prediction, dừng trước khi đụng tới Kaggle API.
from pathlib import Path
from kaggle_shards import (
    validate_predictions,
    build_manifest,
    build_dataset_metadata,
)

_prediction_dir = Path(PART_OUTPUT)
_missing_files, _extra_files = validate_predictions(SELECTED_KEYS, _prediction_dir)
if _missing_files or _extra_files:
    print(">> MISSING predictions:", sorted(_missing_files))
    print(">> EXTRA predictions  :", sorted(_extra_files))
    raise RuntimeError(
        f"Part {RANGE_TAG} chưa hợp lệ: thiếu {len(_missing_files)}, "
        f"thừa {len(_extra_files)} file prediction; chưa upload."
    )

DATASET_SLUG = f"{DATASET_SLUG_PREFIX}-{RANGE_TAG}"
DATASET_TITLE = f"DiffV2IR Generated {RANGE_TAG}"
from kaggle_secrets import UserSecretsClient
_user_secrets = UserSecretsClient()
KAGGLE_USERNAME = _user_secrets.get_secret(KAGGLE_USERNAME_SECRET).strip()
KAGGLE_API_KEY = _user_secrets.get_secret(KAGGLE_KEY_SECRET).strip()
_metadata = build_dataset_metadata(
    username=KAGGLE_USERNAME,
    slug=DATASET_SLUG,
    title=DATASET_TITLE,
)
_manifest = build_manifest(
    source_val_txt=ALL_KEYS_TXT,
    total_source_keys=len(ALL_KEYS),
    start_index=START_INDEX,
    end_index=END_INDEX,
    selected_keys=SELECTED_KEYS,
    generated_prediction_count=len(SELECTED_KEYS),
    missing_prediction_keys=sorted(_missing_files),
    extra_prediction_files=sorted(_extra_files),
    inference_config={
        "resolution": RESOLUTION,
        "steps": STEPS,
        "cfg_text": CFG_TEXT,
        "cfg_image": CFG_IMAGE,
        "cfg_seg": CFG_SEG,
        "seed": SEED,
        "blip_model": BLIP_MODEL,
        "clip_model": CLIP_MODEL,
    },
)
with open(os.path.join(PACKAGE_DIR, "manifest.json"), "w", encoding="utf-8") as _f:
    json.dump(_manifest, _f, indent=2, ensure_ascii=False)
with open(os.path.join(PACKAGE_DIR, "dataset-metadata.json"), "w", encoding="utf-8") as _f:
    json.dump(_metadata, _f, indent=2, ensure_ascii=False)

print(f">> PASS: {len(SELECTED_KEYS)} prediction files khớp chính xác.")
print(">> Manifest:", os.path.join(PACKAGE_DIR, "manifest.json"))
print(">> Dataset slug dự kiến:", DATASET_SLUG)

# %%
# ===================== CELL 8: ĐĂNG NHẬP VÀ TẠO DATASET PRIVATE =====================
# Chỉ chạy sau CELL 7. Kaggle mặc định tạo dataset private khi không truyền cờ công khai.
# Nếu slug đã tồn tại, Kaggle sẽ báo lỗi; hãy đổi DATASET_SLUG_PREFIX hoặc dùng dataset update.
from kaggle_shards import write_kaggle_credentials

write_kaggle_credentials(
    KAGGLE_USERNAME,
    KAGGLE_API_KEY,
    Path("/root/.kaggle/kaggle.json"),
)
DATASET_ID = f"{KAGGLE_USERNAME}/{DATASET_SLUG}"

sh(f"kaggle datasets create -p {PACKAGE_DIR} --dir-mode zip")
print(">> Kaggle Dataset private:", f"https://www.kaggle.com/datasets/{DATASET_ID}")
sh(f"kaggle datasets status {DATASET_ID}")
