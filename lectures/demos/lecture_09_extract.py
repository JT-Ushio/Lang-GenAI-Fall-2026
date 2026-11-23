"""Optional real MP4 decoder/extractor; FFmpeg is not required for normal trace compilation."""
from pathlib import Path
import argparse, json, os, shutil, subprocess
import numpy as np
from PIL import Image


def find_ffmpeg():
    executable=os.environ.get('LECTURE09_FFMPEG') or shutil.which('ffmpeg')
    if executable:return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError('Install FFmpeg or imageio-ffmpeg for video extraction.') from exc


def decode_teaching_clip(path, width=160, height=90):
    # Our fixture is constant 8 FPS; arbitrary VFR videos require their own PTS.
    command=[find_ffmpeg(),'-hide_banner','-loglevel','error','-i',str(path),
             '-an','-vf',f'scale={width}:{height}', '-fps_mode','passthrough',
             '-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    raw=subprocess.run(command,check=True,stdout=subprocess.PIPE).stdout
    frames=np.frombuffer(raw,dtype=np.uint8).reshape(-1,height,width,3).copy()
    return frames


def main():
    parser=argparse.ArgumentParser(description='Extract uniformly spaced frames from a CFR video.')
    parser.add_argument('input',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--count',type=int,default=8)
    parser.add_argument('--source-fps',type=float,required=True,help='Known CFR source rate; not an arbitrary VFR estimate')
    args=parser.parse_args();frames=decode_teaching_clip(args.input)
    if not 1<=args.count<=len(frames) or args.source_fps<=0:parser.error('Invalid count or source FPS')
    ids=np.floor((np.arange(args.count)+.5)*len(frames)/args.count).astype(int)
    args.output.mkdir(parents=True,exist_ok=True)
    for i in ids:Image.fromarray(frames[i]).save(args.output/f'frame_{i:04d}.png')
    info=[{'frame':int(i),'seconds':float(i/args.source_fps)} for i in ids]
    (args.output/'frames.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))


if __name__=='__main__':main()
