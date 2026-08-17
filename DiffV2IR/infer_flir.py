# -*- coding: utf-8 -*-
"""
DiffV2IR inference + evaluation cho dataset FLIR (chạy được trên Kaggle)
========================================================================
- Đọc danh sách ảnh validation từ align_validation.txt
- Với mỗi key: tìm ảnh RGB + seg map, chạy DiffV2IR sinh ảnh IR dự đoán
- So sánh với ảnh IR gốc (ground truth .jpeg) theo PSNR / SSIM / FID / LPIPS

Cấu trúc dataset FLIR `align/`:
    AnnotatedImages/  Annotations/  JPEGImages/  align_train.txt  align_validation.txt

Quy ước tên file (FLIR):
    JPEGImages/FLIR_00002_PreviewData.jpeg   <- ảnh IR gốc (ground truth cho metrics)
    JPEGImages/FLIR_00002_RGB.jpg            <- ảnh RGB
    seg/FLIR_00002_RGB.png                   <- seg map (tên GIỐNG HỆT ảnh RGB, đuôi .png)
    Annotations/FLIR_00002_PreviewData.xml   <- annotation object detection (KHÔNG dùng cho metrics ảnh)

Với mỗi key `FLIR_00002_PreviewData` trong align_validation.txt:
    prefix = "FLIR_00002"            (bỏ phần "_PreviewData")
    RGB    = <input-rgb>/FLIR_00002_RGB.jpg
    seg    = <seg-dir>/FLIR_00002_RGB.png
    GT IR  = <gt-dir>/FLIR_00002_PreviewData.jpeg

Toàn bộ đường dẫn đều truyền qua tham số dòng lệnh (phù hợp Kaggle):

    !python infer_flir.py \
        --input-rgb  /kaggle/input/flir/align/JPEGImages \
        --seg-dir    /kaggle/input/flir/align/seg \
        --val-txt    /kaggle/input/flir/align/align_validation.txt \
        --gt-dir     /kaggle/input/flir/align/JPEGImages \
        --config     configs/generate.yaml \
        --ckpt       /kaggle/input/diffv2ir/diffv2ir.ckpt \
        --output     /kaggle/working/out
"""

import argparse
import json
import math
import os
import sys
from contextlib import nullcontext

import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps

sys.path.append("./")                 # để import được stable_diffusion từ DiffV2IR/
sys.path.append("./stable_diffusion") # để import được ldm.*

from stable_diffusion.ldm.util import instantiate_from_config

# Cờ ghi nhận checkpoint có chứa trọng số EMA hay không (đặt trong load_model_from_config)
_HAS_EMA = False


def parse_args():
    p = argparse.ArgumentParser(
        description="DiffV2IR inference + eval trên FLIR (path đều truyền qua CLI)")

    # ---- model ----
    p.add_argument("--config", required=True,
                   help="path tới configs/generate.yaml")
    p.add_argument("--ckpt", required=True,
                   help="path tới DiffV2IR checkpoint (.ckpt)")
    p.add_argument("--vae-ckpt", default=None,
                   help="(tùy chọn) path tới VAE riêng nếu ckpt không đủ first_stage_model")

    # ---- dữ liệu ----
    p.add_argument("--input-rgb", required=True,
                   help="folder chứa ảnh RGB gốc (các file *_RGB.jpg)")
    p.add_argument("--seg-dir", required=True,
                   help="folder chứa seg map (tên file trùng ảnh RGB, đuôi .png)")
    p.add_argument("--val-txt", required=True,
                   help="path tới align_validation.txt (1 key mỗi dòng, dạng FLIR_xxxxx_PreviewData)")
    p.add_argument("--gt-dir", default=None,
                   help="folder chứa ảnh IR gốc *.jpeg (mặc định = input-rgb)")
    p.add_argument("--output", required=True,
                   help="folder lưu ảnh IR dự đoán + file báo cáo metrics")

    # ---- tham số sinh ảnh ----
    p.add_argument("--resolution", type=int, default=512,
                   help="cạnh dài sau resize (mặc định 512)")
    p.add_argument("--steps", type=int, default=100,
                   help="số bước sampling ODE (mặc định 100)")
    p.add_argument("--cfg-text", type=float, default=7.5,
                   help="hệ số CFG cho text (mặc định 7.5)")
    p.add_argument("--cfg-image", type=float, default=1.5,
                   help="hệ số CFG cho ảnh RGB (mặc định 1.5)")
    p.add_argument("--cfg-seg", type=float, default=1.5,
                   help="hệ số CFG cho seg (mặc định 1.5)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed cho sampling (mặc định 0)")

    # ---- BLIP caption ----
    p.add_argument("--no-blip", action="store_true",
                   help="bỏ bước BLIP caption, dùng prompt mặc định '--prompt'")
    p.add_argument("--prompt", default="turn the visible image into infrared",
                   help="prompt dùng khi --no-blip (mặc định: biến visible thành infrared)")
    p.add_argument("--blip-model", default="Salesforce/blip-image-captioning-base",
                   help="model BLIP caption trên HF Hub (transformers tự tải weights, "
                        "không cần file .pth — link GCS gốc của BLIP đã bị Salesforce khóa)")
    p.add_argument("--clip-version", default="openai/clip-vit-large-patch14",
                   help="tên model CLIP trên HuggingFace Hub (mặc định tải CLIP ViT-L/14)")

    # ---- giới hạn (chạy thử nhanh) ----
    p.add_argument("--limit", type=int, default=0,
                   help="chỉ xử lý N ảnh đầu (0 = tất cả)")
    p.add_argument("--metrics-only", action="store_true",
                   help="bỏ qua sampling, chỉ tính metrics từ các file pred đã có trong --output")
    p.add_argument("--fid-min-images", type=int, default=50,
                   help="số ảnh tối thiểu để FID có ý nghĩa thống kê (dưới ngưỡng -> FID=nan, "
                        "mặc định 50)")

    # ---- visualization + logging ----
    p.add_argument("--vis-num", type=int, default=20,
                   help="số ảnh dùng cho visualization (mặc định 20)")
    p.add_argument("--vis-folder", default=None,
                   help="folder lưu ảnh visualization (mặc định: <output>/visualization)")
    p.add_argument("--wandb", action="store_true",
                   help="bật logging lên wandb (cần set biến môi trường WANDB_API_KEY)")
    p.add_argument("--wandb-project", default="diffv2ir-flir",
                   help="tên project wandb (mặc định diffv2ir-flir)")
    p.add_argument("--wandb-run-name", default=None,
                   help="tên run wandb (mặc định: tự sinh theo thời gian)")

    return p.parse_args()


# ============================== BƯỚC 2: NẠP MODEL ==============================


def _patch_clip_version(version):
    """Sửa path mặc định của FrozenCLIPEmbedder.

    `stable_diffusion/ldm/modules/encoders/modules.py:139` hardcode version =
    "/data/wld/ip2p/clip-vit-large-patch14/" — đường dẫn này KHÔNG tồn tại trên
    Kaggle/máy khác. Ta override default để transformers tự tải `openai/clip-vit-...`
    từ HuggingFace Hub. Gọi TRƯỚC khi load model từ config."""
    # LƯU Ý: phải import qua namespace `ldm.modules.encoders.modules` (KHÔNG phải
    # `stable_diffusion.ldm...`) vì config trong generate.yaml dùng target
    # `ldm.modules.encoders.modules.FrozenCLIPEmbedder` (get_obj_from_str import
    # theo tên `ldm.*`). Do infer_flir.py thêm "./stable_diffusion" vào sys.path,
    # Python coi 2 tên đó là 2 module riêng — patch nhầm bản kia thì không ăn.
    from ldm.modules.encoders.modules import FrozenCLIPEmbedder

    orig_init = FrozenCLIPEmbedder.__init__

    def patched(self, *args, **kwargs):
        kwargs.setdefault("version", version)
        orig_init(self, *args, **kwargs)

    FrozenCLIPEmbedder.__init__ = patched


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    """Khởi tạo model từ config rồi nạp trọng số .ckpt.

    Bản copy sát logic tác giả (infer.py:62-83). Tách riêng để:
    - Dùng lại cho cả lúc load model chính.
    - `strict=False` cho phép bỏ qua `model_ema.*` nếu ckpt thiếu (không lỗi)."""
    print(f">> Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f">> Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f">> Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model."):]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:", m)
    if len(u) > 0 and verbose:
        print("unexpected keys:", u)
    return model


class CFGDenoiser(nn.Module):
    """4-way Classifier-Free Guidance — khớp chính xác infer.py:35-49.

    DiffV2IR điều kiện hóa bởi 3 kênh: text (c_crossattn), RGB latent (c_concat1),
    seg latent (c_concat2). Với mỗi bước denoise, ta chạy 1 batch 4 ảnh giống nhau:
        1. cond       : có text + có RGB + có seg
        2. img_cond   : có text + có RGB + KHÔNG seg
        3. seg_cond   : có text + KHÔNG RGB + có seg
        4. uncond     : không gì cả (text rỗng + zero latent)
    rồi kết hợp theo công thức gradient CFG cho 3 nguồn điều khiển."""
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale, seg_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=4)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=4)
        cfg_cond = {
            "c_crossattn": [torch.cat([
                cond["c_crossattn"][0], uncond["c_crossattn"][0],
                uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat1": [torch.cat([
                cond["c_concat1"][0], cond["c_concat1"][0],
                uncond["c_concat1"][0], uncond["c_concat1"][0]])],
            "c_concat2": [torch.cat([
                cond["c_concat2"][0], cond["c_concat2"][0],
                cond["c_concat2"][0], uncond["c_concat2"][0]])],
        }
        out_cond, out_img_cond, out_seg_cond, out_uncond = \
            self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(4)
        return (out_uncond
                + text_cfg_scale * (out_cond - out_img_cond)
                + image_cfg_scale * (out_img_cond - out_seg_cond)
                + seg_cfg_scale * (out_seg_cond - out_uncond))


class DiffV2IRModel:
    """Bọc toàn bộ DiffV2IR để Bước 3+ chỉ cần gọi model.encode / model.sample."""
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f">> Device: {self.device}")

        _patch_clip_version(args.clip_version)   # sửa hardcoded CLIP path

        config = OmegaConf.load(args.config)
        self.model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
        self.model.eval().to(self.device)

        # bọc model bằng k_diffusion: cung cấp sigma schedule + hàm denoise
        self.model_wrap = K.external.CompVisDenoiser(self.model)
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)

        # null token = embedding của chuỗi rỗng, dùng cho nhánh "uncond" trong CFG
        with torch.no_grad():
            self.null_token = self.model.get_learned_conditioning([""])

        # BLIP caption (tùy chọn) — dùng ảnh RGB để tạo prompt mô tả nội dung.
        # Dùng transformers BlipForConditionalGeneration (cùng model BLIP base caption
        # COCO của bài báo, bản HF chính thức) vì link GCS chứa model_base_caption.pth
        # gốc đã bị Salesforce khóa (403) từ năm 2024. Transformers tự tải weights.
        self.blip = None
        self.blip_processor = None
        if not args.no_blip:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            print(f">> Loading BLIP captioner from {args.blip_model}")
            self.blip_processor = BlipProcessor.from_pretrained(args.blip_model)
            self.blip = BlipForConditionalGeneration.from_pretrained(
                args.blip_model).to(self.device)
            self.blip.eval()

    # ---- tiền xử lý ----
    def load_input(self, pil_img):
        """PIL RGB [0,255] -> tensor [-1,1] 1x3xHxW (giống infer.py:150-153)."""
        x = 2 * torch.tensor(np.array(pil_img)).float() / 255 - 1
        return rearrange(x, "h w c -> 1 c h w").to(self.device)

    # ---- encode / caption / sample ----
    @torch.no_grad()
    def get_latents(self, rgb_pil, seg_pil):
        """RGB + seg -> 2 latent 4 kênh (encode_first_stage), dùng cho c_concat1/2."""
        rgb = self.load_input(rgb_pil)
        seg = self.load_input(seg_pil)
        z_rgb = self.model.encode_first_stage(rgb).mode()
        z_seg = self.model.encode_first_stage(seg).mode()
        return z_rgb, z_seg

    @torch.no_grad()
    def get_prompt(self, rgb_pil):
        """Tạo prompt: BLIP caption nếu bật, ngược lại dùng --prompt."""
        if self.blip is None:
            return self.args.prompt
        inputs = self.blip_processor(rgb_pil, return_tensors="pt").to(self.device)
        out = self.blip.generate(**inputs, do_sample=True, top_p=0.9,
                                 max_length=20, min_length=5)
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return f"turn the visible image of {caption} into infrared"

    @torch.no_grad()
    def sample(self, prompt, z_rgb, z_seg, seed):
        """Sampling euler_ancestral qua CFGDenoiser -> trả về PIL ảnh IR."""
        cond = {
            "c_crossattn": [self.model.get_learned_conditioning([prompt])],
            "c_concat1": [z_rgb],
            "c_concat2": [z_seg],
        }
        uncond = {
            "c_crossattn": [self.null_token],
            "c_concat1": [torch.zeros_like(z_rgb)],
            "c_concat2": [torch.zeros_like(z_seg)],
        }
        sigmas = self.model_wrap.get_sigmas(self.args.steps)
        extra_args = {
            "cond": cond, "uncond": uncond,
            "text_cfg_scale": self.args.cfg_text,
            "image_cfg_scale": self.args.cfg_image,
            "seg_cfg_scale": self.args.cfg_seg,
        }
        # In tiến trình mỗi 10 bước + bước cuối, để biết GPU còn sống và đang ở đâu.
        def _progress(info):
            i = info["i"]
            if i % 10 == 0 or i == self.args.steps - 1:
                print(f"    step {i+1}/{self.args.steps}  sigma={float(info['sigma']):.3f}", flush=True)

        torch.manual_seed(seed)
        z = torch.randn_like(z_rgb) * sigmas[0]
        with torch.autocast("cuda", enabled=self.device == "cuda"):
            z = K.sampling.sample_euler_ancestral(
                self.model_wrap_cfg, z, sigmas,
                extra_args=extra_args, callback=_progress)
            x = self.model.decode_first_stage(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "1 c h w -> h w c")
        return Image.fromarray(x.type(torch.uint8).cpu().numpy())


# ============================== BƯỚC 3: INFERENCE 1 ẢNH ==============================


def resize_fit(img, resolution):
    """Resize ảnh sao cho cạnh dài = resolution, W/H là bội của 64.

    Copy logic infer.py:136-141. BẮT BUỘC vì latent của VAE có stride 8
    (ảnh 512 -> latent 64x64) và UNet dùng attention_resolutions [4,2,1],
    nên kích thước ảnh phải chia hết cho 64, nếu không encode sẽ lỗi kích thước."""
    width, height = img.size
    factor = resolution / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    return ImageOps.fit(img, (width, height),
                        method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def _first_existing(directory, *candidates):
    """Trả về path đầu tiên tồn tại trong `directory`; nếu không có thì trả về
    candidate đầu tiên (để bước gọi kiểm tra isfile và báo 'thiếu file' rõ ràng)."""
    for name in candidates:
        p = os.path.join(directory, name)
        if os.path.isfile(p):
            return p
    return os.path.join(directory, candidates[0])


def key_to_paths(key, args):
    """key `FLIR_xxxxx_PreviewData` -> (rgb_path, seg_path, gt_path).

    Tự dò quy ước tên file THẬT trong dataset — FLIR trên Kaggle có nhiều layout
    khác nhau (RGB .jpg/.jpeg, seg .png/.jpg, tên _RGB/_PreviewData) nên không
    hardcode 1 quy ước. Thứ tự ưu tiên:
        prefix = key bỏ phần "_PreviewData" (vd "FLIR_00002")
        RGB    = {prefix}_RGB.{jpg,jpeg,png} | {prefix}.{jpg,jpeg}
        seg    = {prefix}_RGB.{png,jpg} | {prefix}_PreviewData.{png,jpg} | {prefix}.png
        GT IR  = {key}.{jpeg,jpg,png} | {prefix}_PreviewData.{jpeg,jpg}
    Trả về path tồn tại (hoặc candidate đầu tiên nếu thiếu — caller tự kiểm tra isfile)."""
    prefix = key[:-len("_PreviewData")] if key.endswith("_PreviewData") else key
    gt_dir = args.gt_dir if args.gt_dir else args.input_rgb

    rgb = _first_existing(
        args.input_rgb,
        f"{prefix}_RGB.jpg", f"{prefix}_RGB.jpeg", f"{prefix}_RGB.png",
        f"{prefix}.jpg", f"{prefix}.jpeg",
    )
    seg = _first_existing(
        args.seg_dir,
        f"{prefix}_RGB.png", f"{prefix}_RGB.jpg",
        f"{prefix}_PreviewData.png", f"{prefix}_PreviewData.jpg",
        f"{prefix}.png",
    )
    gt = _first_existing(
        gt_dir,
        f"{key}.jpeg", f"{key}.jpg", f"{key}.png",
        f"{prefix}_PreviewData.jpeg", f"{prefix}_PreviewData.jpg",
    )
    return rgb, seg, gt


def infer_one(model, rgb_path, seg_path, out_path, seed):
    """Đọc 1 cặp (RGB, seg) từ disk -> sinh IR -> lưu file.

    Trả về (rgb_pil, ir_pil) để bước metrics có thể dùng lại nếu cần."""
    rgb = Image.open(rgb_path).convert("RGB")
    seg = Image.open(seg_path).convert("RGB")

    # resize cả hai về cùng kích thước (chia hết 64) rồi encode latent + sample
    rgb = resize_fit(rgb, model.args.resolution)
    seg = resize_fit(seg, model.args.resolution)
    z_rgb, z_seg = model.get_latents(rgb, seg)

    prompt = model.get_prompt(rgb)
    print(f"    prompt : {prompt}")

    ir = model.sample(prompt, z_rgb, z_seg, seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ir.save(out_path)
    return rgb, ir


# ============================== BƯỚC 5: METRICS ==============================


def _pil_to_metric_tensor(pil_img):
    """PIL [0,255] -> tensor [0,1] 1x3xHxW — đúng định dạng torchmetrics.

    Khớp cách ThermalGen đưa dữ liệu vào metrics (thermalgen.py:748-753):
    ảnh trong [0,1], 3 kênh, batch = 1."""
    x = torch.tensor(np.array(pil_img.convert("RGB"))).float() / 255.0
    return rearrange(x, "h w c -> 1 c h w")


def compute_metrics(keys, output_dir, args, device="cuda"):
    """Tính PSNR/SSIM/LPIPS (trung bình từng ảnh) + FID (trên cả tập).

    Đọc cặp ảnh dự đoán `{key}_pred.png` và GT `{key}.jpeg`, resize GT về
    đúng kích thước ảnh dự đoán rồi so sánh — cùng logic ThermalGen/utils/metrics.py:
        PSNR/SSIM : data_range=1.0
        LPIPS     : net alex, normalize=True
        FID       : normalize=True, grayscale -> repeat 3 kênh
    Trả về dict metrics và ghi file `metrics_report.txt` vào output_dir."""
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchmetrics.image.fid import FrechetInceptionDistance

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    psnr_m = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    # FID cần package `torch-fidelity` (torchmetrics chỉ là wrapper). Nếu chưa cài
    # thì để fid_m=None -> bỏ qua FID thay vì crash toàn bộ pipeline.
    fid_m = None
    try:
        fid_m = FrechetInceptionDistance(normalize=True).to(device)
    except ModuleNotFoundError as e:
        print(f"!! Không cài được FID metric: {e}. FID sẽ bỏ qua (nan). "
              f"Cài 'pip install torch-fidelity' để tính FID.")

    n = 0
    missing = []
    for key in keys:
        pred_path = os.path.join(output_dir, f"{key}_pred.png")
        _, _, gt_path = key_to_paths(key, args)
        if not (os.path.isfile(pred_path) and os.path.isfile(gt_path)):
            missing.append(key)
            print(f"!! [{key}] thiếu pred hoặc GT, bỏ qua")
            continue

        pred = _pil_to_metric_tensor(Image.open(pred_path)).to(device)

        # GT: CẮT theo đúng khung pred (giống phép crop model áp lên RGB trong
        # resize_fit: ImageOps.fit centering=0.5) thay vì kéo giãn trơn.
        # -> GT, pred cùng khung tọa độ, tránh "lệch" do crop khác resize.
        gt_pil = Image.open(gt_path).convert("RGB")
        if gt_pil.size != (pred.shape[3], pred.shape[2]):
            gt_pil = ImageOps.fit(gt_pil, (pred.shape[3], pred.shape[2]),
                                  method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        gt = _pil_to_metric_tensor(gt_pil).to(device)

        # PSNR/SSIM/LPIPS: so sánh từng ảnh
        psnr_m.update(pred, gt)
        ssim_m.update(pred, gt)
        lpips_m.update(pred, gt)

        # FID: gom toàn tập rồi tính (pred = "generated", gt = "real")
        # Chỉ update khi FID được khởi tạo được (tức có torch-fidelity).
        if fid_m is not None:
            if pred.shape[1] == 1:
                fid_m.update(pred.repeat(1, 3, 1, 1), real=False)
            else:
                fid_m.update(pred, real=False)
            if gt.shape[1] == 1:
                fid_m.update(gt.repeat(1, 3, 1, 1), real=True)
            else:
                fid_m.update(gt, real=True)
        n += 1

    if n == 0:
        print(">> Không có ảnh nào để tính metrics.")
        return {}

    psnr = psnr_m.compute().item()
    ssim = ssim_m.compute().item()
    lpips = lpips_m.compute().item()

    # FID là metric theo phân phối: cần ĐỦ nhiều ảnh để ước lượng covariance
    # 2048 chiều ổn định. Ngưỡng mặc định 50 (từ --fid-min-images). N quá nhỏ
    # (vd --limit 5 chỉ để test) thì FID vô nghĩa -> để nan, không gây hiểu nhầm.
    if fid_m is None:
        fid = float("nan")
        fid_str = "nan (chưa cài torch-fidelity)"
    elif n >= args.fid_min_images:
        fid = fid_m.compute().item()
        fid_str = f"{fid:.4f}"
    else:
        fid = float("nan")
        fid_str = "nan (cần >= {} ảnh, hiện {} ảnh)".format(args.fid_min_images, n)

    metrics = {"PSNR": psnr, "SSIM": ssim, "LPIPS": lpips, "FID": fid, "N": n}

    print("\n" + "=" * 46)
    print("          KẾT QUẢ METRICS")
    print("=" * 46)
    print(f"   PSNR  : {psnr:.4f}   (cao hơn tốt hơn)")
    print(f"   SSIM  : {ssim:.4f}   (cao hơn tốt hơn)")
    print(f"   LPIPS : {lpips:.4f}   (thấp hơn tốt hơn)")
    print(f"   FID   : {fid_str}   (thấp hơn tốt hơn)")
    print(f"   N     : {n} ảnh so sánh")
    print("=" * 46)

    report_path = os.path.join(output_dir, "metrics_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("DiffV2IR inference on FLIR — metrics report\n")
        f.write(f"N = {n} images\n\n")
        for name, val in metrics.items():
            if name != "N":
                f.write(f"{name}: {val:.6f}\n")
        if missing:
            f.write("\nSkipped (missing pred/gt):\n" + "\n".join(f"  {k}" for k in missing) + "\n")
    print(f">> Đã ghi báo cáo: {report_path}")

    # ---- BƯỚC 6.2: metrics.json — máy đọc được, kèm config để tái hiện thí nghiệm ----
    def _json_safe(x):
        # FID có thể = nan khi N < ngưỡng; JSON strict không cho NaN -> chuyển thành null
        return None if isinstance(x, float) and math.isnan(x) else x

    json_payload = {
        "metrics": {k: _json_safe(v) for k, v in metrics.items() if k != "N"},
        "n_images": n,
        "config": dict(vars(args)),    # toàn bộ tham số dòng lệnh (Namespace -> dict)
        "skipped_missing": missing,
    }
    json_path = os.path.join(output_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)
    print(f">> Đã ghi JSON: {json_path}")
    return metrics


# ===================== BƯỚC 6.1: VISUALIZATION =====================


def build_visualization(keys, args):
    """Tạo ảnh so sánh trực quan: RGB | IR GT | IR Generated + index.

    - Lọc các key có đủ 3 file (RGB, GT, pred), rồi chọn `--vis-num` ảnh
      DÀN ĐỀU trong danh sách đó (phủ toàn bộ validation set, deterministic).
    - Grid panel: mỗi panel 5 ảnh, mỗi dòng = [index, RGB, IR GT, IR Gen].
    - Triplet: mỗi ảnh 1 strip ngang [RGB | IR GT | IR Gen] lưu riêng 1 folder.
    - Trả về list (tên, PIL) của các panel để Bước 6.3 log lên wandb."""
    import matplotlib
    matplotlib.use("Agg")                      # không cần cửa sổ GUI (Kaggle)
    import matplotlib.pyplot as plt

    vis_folder = args.vis_folder or os.path.join(args.output, "visualization")
    triplets_dir = os.path.join(vis_folder, "triplets")
    os.makedirs(triplets_dir, exist_ok=True)

    # B1: lọc toàn bộ key có đủ 3 file (RGB, GT, pred)
    valid_keys = []
    for key in keys:
        rgb_path, seg_path, gt_path = key_to_paths(key, args)
        pred_path = os.path.join(args.output, f"{key}_pred.png")
        if (os.path.isfile(rgb_path) and os.path.isfile(gt_path)
                and os.path.isfile(pred_path)):
            valid_keys.append(key)

    # B2: chọn `--vis-num` ảnh DÀN ĐỀU trong danh sách hợp lệ (phủ toàn bộ validation set)
    m = len(valid_keys)
    if m <= args.vis_num:
        chosen = valid_keys               # ít hơn số cần -> lấy hết
    else:
        step = m / args.vis_num
        chosen = [valid_keys[int(i * step)] for i in range(args.vis_num)]

    # B3: load ảnh của các key đã chọn
    rows = []  # (index, rgb, gt, pred)
    for key in chosen:
        rgb_path, seg_path, gt_path = key_to_paths(key, args)
        pred_path = os.path.join(args.output, f"{key}_pred.png")
        pred = Image.open(pred_path).convert("RGB")

        # RGB: tái lập ĐÚNG phép crop model đã dùng (resize_fit: ImageOps.fit center)
        rgb = Image.open(rgb_path).convert("RGB")
        if rgb.size != pred.size:
            rgb = resize_fit(rgb, args.resolution)

        # GT: CẮT theo đúng khung pred thay vì kéo giãn trơn -> cùng tọa độ với pred
        gt = Image.open(gt_path).convert("RGB")
        if gt.size != pred.size:
            gt = ImageOps.fit(gt, pred.size,
                              method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

        prefix = key[:-len("_PreviewData")] if key.endswith("_PreviewData") else key
        idx = f"{int(prefix.split('_')[-1]):04d}"   # FLIR_00002 -> "0002"
        rows.append((idx, rgb, gt, pred))

    if not rows:
        print(">> Không có ảnh nào đủ (RGB+GT+pred) để visualization.")
        return []

    cols = ["Index", "RGB", "IR GT", "IR Generated"]
    per_panel = 5
    panels = []
    for pi in range(0, len(rows), per_panel):
        chunk = rows[pi:pi + per_panel]
        fig, axes = plt.subplots(len(chunk), len(cols),
                                 figsize=(4 * len(cols), 4 * len(chunk)))
        axes = np.atleast_2d(axes)               # chống lỗi khi chỉ 1 dòng
        for r, (idx, rgb, gt, pred) in enumerate(chunk):
            axes[r, 0].text(0.5, 0.5, idx, ha="center", va="center",
                            fontsize=20, fontweight="bold")
            axes[r, 0].axis("off")
            for c, img in enumerate([rgb, gt, pred], start=1):
                axes[r, c].imshow(np.array(img))
                axes[r, c].axis("off")
        for c, title in enumerate(cols):
            axes[0, c].set_title(title, fontsize=14)
        fig.tight_layout()

        panel_name = f"panel_{pi // per_panel + 1:02d}"
        panel_path = os.path.join(vis_folder, panel_name + ".png")
        fig.savefig(panel_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        panels.append((panel_name, Image.open(panel_path)))

    # 20 ảnh triplet: mỗi ảnh là strip ngang [RGB | IR GT | IR Generated]
    for idx, rgb, gt, pred in rows:
        strip = Image.new("RGB", (pred.width * 3, pred.height))
        strip.paste(rgb, (0, 0))
        strip.paste(gt, (pred.width, 0))
        strip.paste(pred, (2 * pred.width, 0))
        strip.save(os.path.join(triplets_dir, f"{idx}_rgb_gt_gen.png"))

    print(f">> Visualization: {len(rows)} ảnh -> {len(panels)} panel, "
          f"triplet lưu tại {triplets_dir}")
    return panels


# ============================== BƯỚC 6.3: WANDB ==============================


def log_to_wandb(args, metrics, vis_panels):
    """Log metrics + panel visualization lên wandb. Chỉ chạy khi `--wandb`.

    Trên Kaggle cần login trước bằng 1 cell:
        !wandb login <API_KEY>      (hoặc set biến env WANDB_API_KEY)
    """
    if not args.wandb:
        print(">> (Bỏ qua wandb — không truyền --wandb)")
        return

    import wandb
    run = wandb.init(project=args.wandb_project,
                     name=args.wandb_run_name,
                     config=dict(vars(args)))
    print(f">> Wandb run: {run.name} (project={args.wandb_project})")

    # metrics dạng bảng trong wandb (các giá trị vô hướng -> chart tự sinh)
    wandb.log(metrics)

    # 4 panel visualization — mỗi panel là 1 ảnh trong wandb, xem được trên web
    for name, pil in vis_panels:
        wandb.log({name: [wandb.Image(pil)]})

    run.finish()
    print(">> Đã đẩy xong dữ liệu lên wandb.")


def main():
    args = parse_args()
    print(">>> Cấu hình nhận được:")
    for k, v in sorted(vars(args).items()):
        print(f"    {k:12} = {v}")

    model = DiffV2IRModel(args)
    print(">>> Model + CFGDenoiser đã sẵn sàng.")

    if args.metrics_only:
        print(">> --metrics-only: bỏ qua sampling, tính metrics từ các ảnh pred đã có.")
    else:
        # ====================== BƯỚC 4: CHẠY TOÀN BỘ VALIDATION SET ======================
        if not os.path.isfile(args.val_txt):
            raise SystemExit(f"Không tìm thấy --val-txt: {args.val_txt}")

        keys = [ln.strip() for ln in open(args.val_txt, encoding="utf-8") if ln.strip()]
        if not keys:
            raise SystemExit(f"--val-txt trống: {args.val_txt}")

        if args.limit:
            keys = keys[:args.limit]
        print(f">> Sẽ xử lý {len(keys)} ảnh (limit={args.limit or 'all'})")

        os.makedirs(args.output, exist_ok=True)
        done, skipped, missing = 0, 0, []
        seed = args.seed

        for i, key in enumerate(keys):
            rgb_path, seg_path, _ = key_to_paths(key, args)

            # --no-skip-missing không có; nếu thiếu RGB/seg ta ghi nhận và bỏ qua
            if not (os.path.isfile(rgb_path) and os.path.isfile(seg_path)):
                missing.append(key)
                print(f"!! [{i+1}/{len(keys)}] thiếu file: {key}")
                continue

            # bỏ qua nếu đã có kết quả (resume khi chạy lại giữa chừng)
            out_path = os.path.join(args.output, f"{key}_pred.png")
            if os.path.isfile(out_path):
                skipped += 1
                print(f"-- [{i+1}/{len(keys)}] {key}: đã có kết quả, bỏ qua")
                continue

            print(f">> [{i+1}/{len(keys)}] {key}: sampling...")
            try:
                _, _ = infer_one(model, rgb_path, seg_path, out_path, seed)
                done += 1
            except Exception as e:
                missing.append(key)
                print(f"!! [{i+1}/{len(keys)}] {key}: lỗi - {e}")

        print(f"\n=== Bước 4 xong: done={done}, skipped={skipped}, missing/failed={len(missing)} ===")
        if missing:
            print("Danh sách thiếu/không xử lý được:", ", ".join(missing))

    # ====================== BƯỚC 5: METRICS ======================
    # Nếu chạy sampling: dùng đúng danh sách key đã xử lý (bị cắt bởi --limit).
    # Nếu --metrics-only: đọc key từ --val-txt (cũng tôn trọng --limit).
    metric_keys = [ln.strip() for ln in open(args.val_txt, encoding="utf-8") if ln.strip()]
    if args.limit:
        metric_keys = metric_keys[:args.limit]

    metrics = compute_metrics(metric_keys, args.output, args,
                              device=model.device if not args.metrics_only else "cuda")

    # ====================== BƯỚC 6.1: VISUALIZATION ======================
    # Tạo panel so sánh (RGB | IR GT | IR Generated) + 20 triplet; trả về panels
    # để Bước 6.3 đẩy lên wandb.
    vis_panels = build_visualization(metric_keys, args)

    # ====================== BƯỚC 6.3: WANDB ======================
    # Log metrics + 4 panel lên wandb (chỉ khi --wandb).
    log_to_wandb(args, metrics, vis_panels)


if __name__ == "__main__":
    main()
