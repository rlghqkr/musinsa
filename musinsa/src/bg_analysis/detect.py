import numpy as np
# 실제 구현에서는 모델 로더/추론 코드 추가

class Detector:
    def __init__(self, cfg):
        self.cfg = cfg
        # TODO: load Grounding-DINO weights or YOLOv8x-seg

    def __call__(self, pil_img):
        # return list of dicts: {"label": str, "bbox": [x1,y1,x2,y2], "score": float}
        # TODO: implement real detection
        w, h = pil_img.size
        return [{"label":"bench","bbox":[0.1*w,0.6*h,0.5*w,0.9*h],"score":0.7}]
