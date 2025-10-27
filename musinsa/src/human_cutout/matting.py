import numpy as np
from PIL import Image

def refine_alpha(pil_img, mask, cfg):
    # 고품질: MODNet/BackgroundMattingV2 연결
    # 여기서는 간단 feather
    from scipy.ndimage import gaussian_filter
    alpha = mask.astype(float)
    alpha = gaussian_filter(alpha, sigma=1.0)
    alpha = alpha / (alpha.max() + 1e-6)
    return alpha
