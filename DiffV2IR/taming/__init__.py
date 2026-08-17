# Vendored from https://github.com/CompVis/taming-transformers (Apache-2.0)
# Chỉ giữ phần tối thiểu mà stable_diffusion cần ở module-load time:
#     autoencoder.py: from taming.modules.vqvae.quantize import VectorQuantizer
# (DiffV2IR dùng AutoencoderKL nên không cần VQModel/losses của taming.)
