import os, json
from .clip_index import ClipIndexer
from .layout_metric import layout_distance

def retrieve_candidates(bg_sig, cfg):
    index_dir = cfg['paths']['index_dir']
    clip = ClipIndexer(index_dir, cfg['models']['clip_model'])
    # 1) 전역 임베딩 topK
    topk = cfg['retrieval']['topk']
    # 실제 구현: bg_sig에서 원본 PIL 이미지를 넣어야 함. 여기서는 샘플 더미로 대체.
    from PIL import Image
    results = clip.search(Image.new('RGB',(512,512)), topk=topk)

    # 2) 레이아웃/조명 점수 합산
    W = cfg['retrieval']['weights']
    ranked = []
    for r in results:
        meta_json = os.path.splitext(r['path'])[0] + '.json'  # 각 스냅에 대응하는 레이아웃 메타
        cand_layout = json.load(open(meta_json,'r')) if os.path.exists(meta_json) else {'boxes':[], 'labels':[]}
        d_layout = layout_distance(bg_sig, cand_layout)
        score = W['clip']*r['dist'] + W['layout']*d_layout  # light 생략(예시)
        ranked.append({**r, 'score': score})
    ranked.sort(key=lambda x: x['score'])
    return ranked[:min(10, len(ranked))]
