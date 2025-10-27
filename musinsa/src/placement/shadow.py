import numpy as np
from PIL import Image, ImageFilter

def add_contact_shadow(comp, fg_mask, bg_sig, cfg):
    comp = comp.copy()
    h, w = fg_mask.shape
    sh = Image.fromarray((fg_mask*255).astype('uint8')).resize(comp.size, Image.BILINEAR)
    # 바닥 근처 얇은 그림자 밴드
    sh = sh.filter(ImageFilter.GaussianBlur(radius=8))
    import PIL.ImageOps as IO
    sh = IO.invert(sh)
    comp = Image.blend(comp, sh.convert('RGB'), alpha=0.15)
    return comp
