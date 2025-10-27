import numpy as np
from PIL import Image

def composite_with_zorder(bg, fg, fg_mask, bg_sig, cfg):
    # TODO: bg_sig['masks']와 깊이로 전후관계 분리합성
    # 여기선 단순 오버
    bg = bg.copy()
    fg_rgba = fg.convert('RGBA')
    A = Image.fromarray((fg_mask*255).astype('uint8'))
    fg_rgba.putalpha(A)
    bg.paste(fg_rgba, (int((bg.size[0]-fg.size[0])//2), int(bg.size[1]*0.65 - fg.size[1])), fg_rgba)
    return bg
