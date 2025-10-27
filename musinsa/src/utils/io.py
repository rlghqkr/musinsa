import os
from PIL import Image

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_image(path):
    im = Image.open(path).convert('RGB')
    return im

def save_image(im, path):
    ensure_dir(os.path.dirname(path))
    im.save(path)
