class Segmenter:
    def __init__(self, cfg):
        self.cfg = cfg
        # TODO: load SAM2

    def masks_from_boxes(self, pil_img, boxes):
        # boxes: list of [x1,y1,x2,y2]
        # return list of binary masks (numpy HxW bool)
        # TODO: call SAM predictor
        w, h = pil_img.size
        import numpy as np
        masks = []
        for b in boxes:
            m = np.zeros((h, w), dtype=bool)
            x1,y1,x2,y2 = map(int,b)
            m[y1:y2, x1:x2] = True
            masks.append(m)
        return masks
