"""Resumable bounded HTTP range downloads for unstable large-file connections."""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import subprocess
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('url');p.add_argument('dest',type=Path);p.add_argument('--workers',type=int,default=8);args=p.parse_args()
    args.dest.parent.mkdir(parents=True,exist_ok=True)
    chunks=Path(str(args.dest)+'.chunks');chunks.mkdir(exist_ok=True)
    probe=chunks/'probe'; headers=chunks/'headers'
    subprocess.run(['curl','-fsSL','--http1.1','--retry','4','--max-time','45','-r','0-0','-D',str(headers),'-o',str(probe),args.url],check=True)
    sizes=re.findall(r'content-range:\s*bytes\s+\d+-\d+/(\d+)',headers.read_text(),re.I)
    if not sizes:raise RuntimeError('Server does not supply Content-Range')
    total=int(sizes[-1]); block=4*1024*1024
    old=Path(str(args.dest)+'.part')
    if old.exists():
        with old.open('rb') as source:
            for start in range(0,min(old.stat().st_size,total),block):
                length=min(block,total-start)
                if start+length>old.stat().st_size:break
                part=chunks/str(start)
                if not part.exists():part.write_bytes(source.read(length))
                else:source.seek(length,1)
    def get(start):
        end=min(start+block,total)-1;part=chunks/str(start)
        if part.exists() and part.stat().st_size==end-start+1:return
        temp=chunks/(str(start)+'.tmp')
        hdr=chunks/(str(start)+'.headers')
        for attempt in range(10):
            result=subprocess.run(['curl','-fsSL','--http1.1','--connect-timeout','10','--max-time','45','-r',f'{start}-{end}','-D',str(hdr),'-o',str(temp),args.url],capture_output=True)
            if result.returncode==0 and temp.stat().st_size==end-start+1 and f'bytes {start}-{end}/{total}' in hdr.read_text():
                temp.rename(part);print('chunk',start,end,flush=True);return
            time.sleep(min(attempt+1,5))
        raise RuntimeError(f'Unable to download range {start}-{end}: {result.stderr.decode()}')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(get,range(0,total,block)))
    combined=Path(str(args.dest)+'.complete')
    with combined.open('wb') as output:
        for start in range(0,total,block):output.write((chunks/str(start)).read_bytes())
    expected=re.findall(r'x-linked-etag:\s*"?([a-f0-9]{64})"?',headers.read_text(),re.I)
    if expected:
        with combined.open('rb') as source:
            digest=hashlib.file_digest(source,'sha256').hexdigest()
        if digest!=expected[-1]:raise ValueError(f'SHA256 mismatch for {args.dest}')
    combined.rename(args.dest)
    print('COMPLETE',args.dest,total,flush=True)

if __name__=='__main__':main()
