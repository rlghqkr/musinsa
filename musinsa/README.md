# High‑Fidelity Background→Snap→Human Insertion

## 1) 준비물
- Python 3.10+
- CUDA GPU 권장 (RTX 3060+)
- 무신사 스냅 로컬 폴더 (연구/내부 용도로만 사용 권장)

## 2) 설치
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

> SAM2, Grounding-DINO, Stable Diffusion Inpainting 가중치는 각 레포 지침에 따라 별도 다운로드가 필요합니다.

## 3) 스냅 인덱스 구축
```bash
python scripts/build_index.py --img_dir /path/to/musinsa_snaps --out_dir data/musinsa_index
```

## 4) 단일 배경 실행
```bash
python app/main.py --bg data/samples/bg_01.jpg
```

## 5) 일괄 실행
```bash
python scripts/batch_infer.py --bg_glob "data/samples/*.jpg"
```

## 6) 정확도 향상 팁
- Detector를 Grounding‑DINO로 교체하고 SAM2로 마스크 정교화
- Depth를 ZoeDepth로 교체하고 바닥평면 추정 추가
- Human Parsing + 매팅 모델(MODNet/BGMattingV2)로 머리카락 경계 보정
- Harmonization에 iHarmony4 연동
- Inpainting에 Stable Diffusion Inpainting + IP‑Adapter + ControlNet(Depth) 연결

## 7) 주의
- 외부 인물/브랜드 이미지 사용에 대한 라이선스/저작권/초상권을 확인하세요.
