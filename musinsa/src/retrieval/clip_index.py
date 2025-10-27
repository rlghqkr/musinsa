import os, faiss, numpy as np, pandas as pd
from PIL import Image
# TODO: load CLIP model via transformers or open_clip

class ClipIndexer:
    def __init__(self, index_dir, clip_model_name):
        self.index_dir = index_dir
        self.clip_model_name = clip_model_name
        self.index_path = os.path.join(index_dir, 'clip.index')
        self.meta_path = os.path.join(index_dir, 'meta.parquet')
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = None
        self.meta = pd.read_parquet(self.meta_path) if os.path.exists(self.meta_path) else None

    def embed(self, pil_img):
        # TODO: real CLIP embedding
        w,h = pil_img.size
        return np.array([w,h,1.0,0.0]).astype('float32')

    def search(self, pil_img, topk=20):
        q = self.embed(pil_img)[None, :]
        if self.index is None or self.meta is None:
            raise RuntimeError('Index not built. Run scripts/build_index.py')
        D, I = self.index.search(q, topk)
        results = []
        for d,i in zip(D[0], I[0]):
            rec = self.meta.iloc[i].to_dict()
            rec['dist'] = float(d)
            results.append(rec)
        return results
