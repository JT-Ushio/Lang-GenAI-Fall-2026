"""Download only inference weights/tokenizers, with explicit mirror and no Xet dependency."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

FILES = {
    'openai/clip-vit-base-patch32': ['config.json', 'preprocessor_config.json', 'pytorch_model.bin'],
    'sentence-transformers/clip-ViT-B-32-multilingual-v1': ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt', '2_Dense/model.safetensors'],
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--endpoint', default='https://hf-mirror.com')
    args = p.parse_args()
    def download(item):
        repo, name = item
        dest = args.root / repo.split('/')[-1] / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            return
        part = dest.with_suffix(dest.suffix + '.part')
        subprocess.run(['curl', '-fL', '--retry', '5', '--connect-timeout', '20', '--speed-limit', '1024', '--speed-time', '45', '-C', '-', '-o', str(part), f'{args.endpoint}/{repo}/resolve/main/{name}'], check=True)
        part.rename(dest)
        print('Downloaded', dest, flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, [(repo, name) for repo, names in FILES.items() for name in names]))

if __name__ == '__main__':
    main()
