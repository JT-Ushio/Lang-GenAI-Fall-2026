"""Fetch only the 461 official COCO test image IDs from the COCO image server."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import subprocess

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    ids=(args.root/'official-task1/image_splits/test_2017_mscoco.txt').read_text().splitlines()
    dest=args.root/'images';dest.mkdir(exist_ok=True)
    def download(row):
        name=row.split('#')[0]
        target=dest/name
        if target.exists():return
        folder=name.split('_')[1]
        part=target.with_suffix('.part')
        for attempt in range(20):
            result=subprocess.run(['curl','-fsSL','--connect-timeout','10','--max-time','25','-C','-','-o',str(part),f'http://images.cocodataset.org/{folder}/{name}'],capture_output=True)
            if result.returncode==0:
                break
        else:
            raise RuntimeError(f'Failed to fetch {name}: {result.stderr.decode()}')
        part.rename(target)
        print(name, flush=True)
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(download,ids))
    print('Downloaded',len(ids),'COCO images',flush=True)

if __name__=='__main__':main()
