import numpy as np

class DepthEstimator:
    def __init__(self, cfg):
        self.cfg = cfg
        # TODO: load MiDaS/ZoeDepth via torch.hub or transformers

    def __call__(self, pil_img):
        w, h = pil_img.size
        # TODO: real depth
        yy, xx = np.mgrid[0:h, 0:w]
        depth = (yy / h).astype(np.float32)  # dummy gradient depth
        return depth
