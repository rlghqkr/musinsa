import argparse, yaml, glob, subprocess

def main(cfg_path, bg_glob):
    with open(cfg_path,'r') as f:
        cfg = yaml.safe_load(f)
    for p in glob.glob(bg_glob):
        subprocess.run(['python','app/main.py','--config',cfg_path,'--bg',p], check=True)

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--bg_glob', required=True)
    args = ap.parse_args()
    main(args.config, args.bg_glob)
