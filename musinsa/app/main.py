import argparse, os, yaml
from src.utils.io import load_image, save_image, ensure_dir
from src.bg_analysis.signature import build_background_signature
from src.retrieval.search import retrieve_candidates
from src.human_cutout.human_seg import extract_person_mask
from src.human_cutout.matting import refine_alpha
from src.placement.scale_pose import estimate_scale_and_place
from src.placement.zorder import composite_with_zorder
from src.placement.shadow import add_contact_shadow
from src.harmonize.color_match import color_transfer
from src.harmonize.harmonizer import harmonize_composite
from src.harmonize.inpaint import diffusion_inpaint_refine
from src.metrics.layout_score import layout_consistency
from src.metrics.blend_quality import seam_score


def run(cfg, bg_path):
    os.makedirs(cfg['paths']['output_dir'], exist_ok=True)
    bg = load_image(bg_path)

    # 1) 배경 시그니처
    bg_sig = build_background_signature(bg, cfg)

    # 2) 스냅 후보 검색
    cands = retrieve_candidates(bg_sig, cfg)

    best_img = None
    best_score = 1e9
    report = []

    for cand in cands:
        snap_img = load_image(cand['path'])

        # 3) 인물 마스크/매팅
        person_mask = extract_person_mask(snap_img, cfg)
        alpha = refine_alpha(snap_img, person_mask, cfg)

        # 4) 배치(스케일, 위치, 회전) + Z-order 합성
        placed, placed_mask = estimate_scale_and_place(snap_img, alpha, bg, bg_sig, cfg)
        comp = composite_with_zorder(bg, placed, placed_mask, bg_sig, cfg)

        # 5) 그림자/색조화/확산 인페인팅 마감
        comp = add_contact_shadow(comp, placed_mask, bg_sig, cfg)
        comp = color_transfer(comp, bg, strength=0.3)
        comp = harmonize_composite(comp, bg, cfg)
        comp = diffusion_inpaint_refine(comp, bg, placed_mask, cfg)

        # 6) 점수 산출
        lscore = layout_consistency(comp, bg_sig)
        sscore = seam_score(comp, placed_mask)
        total = 0.6*lscore + 0.4*sscore

        report.append((cand['path'], lscore, sscore, total))
        if total < best_score:
            best_score = total
            best_img = comp

    # 저장
    out_path = os.path.join(cfg['paths']['output_dir'], os.path.basename(bg_path).replace('.', '_out.'))
    out_path = out_path[:-1] + 'png'
    save_image(best_img, out_path)

    # 리포트 저장
    with open(os.path.join(cfg['paths']['output_dir'], 'report.txt'), 'w', encoding='utf-8') as f:
        for p, l, s, t in sorted(report, key=lambda x: x[3]):
            f.write(f"{p}\tlayout={l:.4f}\tseam={s:.4f}\ttotal={t:.4f}\n")

    print("Saved:", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--bg', required=True, help='인물 없는 배경 이미지 경로')
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    run(cfg, args.bg)
