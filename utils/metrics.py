import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func

def calculate_metrics(img1, img2):
    # 输入应为 [0, 1] 的 numpy HWC
    psnr = psnr_func(img1, img2, data_range=1.0)
    ssim = ssim_func(img1, img2, data_range=1.0, channel_axis=2)
    return psnr, ssim