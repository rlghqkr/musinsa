import numpy as np
from PIL import Image

def paste_with_alpha(bg, fg, alpha, xy):
    bg = bg.copy()
    fg = fg.convert('RGBA')
    import numpy as np
    a = (np.clip(alpha,0,1)*255).astype('uint8')
    A = Image.fromarray(a)
    fg.putalpha(A)
    bg.paste(fg, xy, fg)
    return bg

def estimate_scale_and_place(snap_img, alpha, bg_img, bg_sig, cfg):
    # TODO: depth-based scale, floor alignment
    bw,bh = bg_img.size
    sw,sh = snap_img.size
    scale = 0.6 * (bh/sh)  # dummy heuristic
    new_size = (int(sw*scale), int(sh*scale))
    fg = snap_img.resize(new_size, Image.LANCZOS)
    import numpy as np
    a = Image.fromarray((np.clip(alpha,0,1)*255).astype('uint8')).resize(new_size, Image.BILINEAR)

    # 배치 위치: 하단 중앙 근처
    x = int((bw - new_size[0]) * 0.5)
    y = int(bh*0.65 - new_size[1])

    return fg, np.array(a)>0
