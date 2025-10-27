from PIL import Image
import numpy as np

def color_transfer(src, ref, strength=0.3):
    # Reinhard color transfer (간이)
    import cv2
    src_np = cv2.cvtColor(np.array(src), cv2.COLOR_RGB2LAB).astype('float32')
    ref_np = cv2.cvtColor(np.array(ref), cv2.COLOR_RGB2LAB).astype('float32')
    ms, ss = src_np.mean(axis=(0,1)), src_np.std(axis=(0,1))+1e-6
    mr, sr = ref_np.mean(axis=(0,1)), ref_np.std(axis=(0,1))+1e-6
    out = (src_np - ms) * (sr/ss) + mr
    out = np.clip(out*(1.0*strength) + cv2.cvtColor(np.array(src), cv2.COLOR_RGB2LAB)*(1-strength), 0, 255)
    out = cv2.cvtColor(out.astype('uint8'), cv2.COLOR_LAB2RGB)
    return Image.fromarray(out)
