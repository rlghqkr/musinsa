import numpy as np
from .detect import Detector
from .segment import Segmenter
from .depth import DepthEstimator

def build_background_signature(pil_img, cfg):
    det = Detector(cfg)
    seg = Segmenter(cfg)
    dep = DepthEstimator(cfg)

    dets = det(pil_img)
    boxes = [d['bbox'] for d in dets]
    labels = [d['label'] for d in dets]
    masks = seg.masks_from_boxes(pil_img, boxes)
    depth = dep(pil_img)

    sig = {
        'boxes': boxes,
        'labels': labels,
        'masks': masks,
        'depth': depth,
        'size': pil_img.size,
    }
    return sig
