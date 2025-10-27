from PIL import Image
import numpy as np

class HumanSeg:
    def __init__(self, cfg):
        self.cfg = cfg
        # TODO: load human parsing or SAM2

    def __call__(self, pil_img):
        # return binary mask (H,W) bool
        w,h = pil_img.size
        m = np.zeros((h,w), dtype=bool)
        m[int(0.2*h):int(0.9*h), int(0.4*w):int(0.6*w)] = True  # dummy
        return m

def extract_person_mask(pil_img, cfg):
    return HumanSeg(cfg)(pil_img)
