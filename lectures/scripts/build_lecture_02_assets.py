"""Rebuild teaching assets; requires numpy, pillow, scipy, matplotlib, sklearn, imageio-ffmpeg.
Run from any directory. On macOS, optional speech is synthesized with `say`.
Checked-in assets let lecture_02 compile without this generator or FFmpeg.
"""
from pathlib import Path
import shutil
import subprocess
import wave

import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import resample_poly, chirp
from sklearn.datasets import load_digits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / 'images/lecture_02'
MEDIA = ROOT / 'videos/lecture_02'
IMG.mkdir(parents=True, exist_ok=True)
MEDIA.mkdir(parents=True, exist_ok=True)
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
plt.rcParams.update({'font.size': 12, 'figure.facecolor': '#fafcfd'})


def run(*args):
    return subprocess.run([FFMPEG, '-hide_banner', '-loglevel', 'error', '-y', *map(str, args)], check=True, capture_output=True).stdout


def wav(path, rate, values):
    with wave.open(str(path), 'wb') as f:
        f.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
        f.writeframes(np.rint(np.clip(values, -1, 1) * 32767).astype('<i2').tobytes())


def savefig(name):
    plt.savefig(IMG / name, dpi=160, bbox_inches='tight', pad_inches=.35)
    plt.close()


def main():
    digits = load_digits()
    digit = digits.images[3].astype(np.uint8)
    np.save(IMG / 'digit.npy', digit)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(digit, cmap='gray', vmin=0, vmax=16, interpolation='nearest')
    ax[0].set_title('Handwritten digit: 3 (8 x 8)')
    ax[1].imshow(digit, cmap='gray', vmin=0, vmax=16)
    for y in range(8):
        for x in range(8):
            ax[1].text(x, y, str(digit[y, x]), ha='center', va='center', color='black' if digit[y,x] > 8 else 'white')
    ax[1].set_title('The same image: 64 intensity values')
    for a in ax:
        a.set_xlabel('column x'); a.set_ylabel('row y')
    savefig('grayscale.png')

    # Original procedural illustration: no external photograph needed.
    canvas = Image.new('RGB', (96, 64), (120, 190, 240))
    d = ImageDraw.Draw(canvas)
    d.rectangle((0, 44, 95, 63), fill=(70, 160, 85))
    d.ellipse((65, 5, 84, 24), fill=(255, 210, 50))
    d.rectangle((25, 31, 59, 53), fill=(245, 225, 180))
    d.polygon([(20, 31), (42, 12), (64, 31)], fill=(210, 50, 45))
    d.rectangle((38, 39, 47, 53), fill=(80, 60, 45))
    canvas.save(IMG / 'rgb.png')
    rgb = np.array(canvas)
    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    ax[0].imshow(rgb); ax[0].set_title('RGB: (64, 96, 3)')
    for i, name in enumerate(['R', 'G', 'B']):
        ax[i+1].imshow(rgb[:,:,i], cmap='gray', vmin=0, vmax=255)
        ax[i+1].set_title(name + ' intensity')
    for a in ax: a.axis('off')
    savefig('rgb-channels.png')
    fig, ax = plt.subplots(1, 3, figsize=(11, 3))
    for a, size in zip(ax, [(96,64), (24,16), (6,4)]):
        a.imshow(canvas.resize(size, Image.Resampling.BOX), interpolation='nearest')
        a.set_title(f'{size[0]} x {size[1]} pixels'); a.axis('off')
    savefig('resolution.png')

    # A clearly labelled synthetic listening stimulus: speech, then a 200–7000 Hz sweep.
    source = Path('/private/tmp/course-speech.aiff')
    if not source.exists():
        if not shutil.which('say'):
            raise RuntimeError('Provide /private/tmp/course-speech.aiff or run on macOS with say.')
        subprocess.run(['say', '-v', 'Samantha', '-r', '145', '-o', str(source), 'She sees six shiny ships. Listen to the same sound at different sample rates.'], check=True)
    speech = np.frombuffer(run('-i', source, '-f', 'f32le', '-ac', 1, '-ar', 48000, 'pipe:1'), dtype='<f4').copy()
    if speech.size == 0:
        raise RuntimeError('Speech file contains no samples; regenerate it with macOS say outside a restricted sandbox.')
    speech *= 0.45 / max(float(np.max(np.abs(speech))), 1e-6)
    t = np.arange(2 * 48000) / 48000
    sweep = 0.18 * chirp(t, f0=200, f1=7000, t1=2) * np.sin(np.pi*t/2)**2
    master = np.concatenate([speech, np.zeros(24000), sweep])
    for rate in [48000, 16000, 8000]:
        values = resample_poly(master, rate // 1000, 48) if rate != 48000 else master
        wav(MEDIA / f'speech_{rate}.wav', rate, values)
    # Show a non-silent segment on a shared time axis.
    start = int(np.argmax(np.abs(speech)) / 48000 * 1000) / 1000
    fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for a, rate in zip(ax, [48000, 16000, 8000]):
        with wave.open(str(MEDIA / f'speech_{rate}.wav')) as f:
            values = np.frombuffer(f.readframes(f.getnframes()), dtype='<i2')
        i = int(start * rate); count = int(.003 * rate)
        a.plot(np.arange(count)/rate*1000, values[i:i+count], '.-', markersize=4)
        a.set_ylabel(f'{rate//1000} kHz\nPCM int16'); a.grid(alpha=.2)
    ax[-1].set_xlabel('Time within the same 3 ms segment (ms)')
    savefig('audio-samples.png')

    # Exactly 48 frames and 96,000 audio samples: two seconds at 24 fps / 48 kHz.
    frames = []
    for i in range(48):
        frame = Image.new('RGB', (160, 96), (235, 244, 250))
        draw = ImageDraw.Draw(frame)
        x = 12 + round(i/47*120)
        draw.ellipse((x-8, 40, x+8, 56), fill=(230, 90, 45))
        draw.text((5, 5), f'frame {i:02d} | {i/24:.3f}s', fill=(20,40,60))
        frames.append(np.array(frame))
    frames = np.array(frames)
    np.save(IMG / 'video-source-frames.npy', frames)
    wav(MEDIA / 'video-tone.wav', 48000, 0.16*np.sin(2*np.pi*440*t)*np.sin(np.pi*t/2)**2)
    process = subprocess.run([FFMPEG, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '160x96', '-r', '24', '-i', 'pipe:0', '-i', str(MEDIA/'video-tone.wav'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', '-c:a', 'aac', '-movflags', '+faststart', '-shortest', str(MEDIA/'motion.mp4')], input=frames.tobytes(), capture_output=True, check=True)
    decoded = np.frombuffer(run('-i', MEDIA/'motion.mp4', '-map', '0:v:0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'), dtype=np.uint8).reshape(-1,96,160,3)
    np.save(IMG / 'video-decoded-frames.npy', decoded)
    run('-i', MEDIA/'motion.mp4', '-map', '0:a:0', '-acodec', 'pcm_s16le', MEDIA/'video-decoded-audio.wav')
    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    for a,i in zip(ax,[0,12,24,36]):
        a.imshow(decoded[i]); a.set_title(f't = {i/24:.1f} s'); a.axis('off')
    savefig('video-frames.png')

    # Points on six cube faces; points have geometry, but no connectivity.
    u,v = np.meshgrid(np.linspace(-1,1,12), np.linspace(-1,1,12))
    p = np.column_stack([u.ravel(),v.ravel()])
    points = np.concatenate([np.insert(p, axis, side, axis=1) for axis in range(3) for side in [-1.,1.]])
    points = np.unique(points, axis=0).astype(np.float32)
    np.save(IMG/'points.npy',points)
    vertices = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]],dtype=np.float32)
    faces = np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]],dtype=np.int32)
    np.savez(IMG/'cube-mesh.npz',vertices=vertices,faces=faces)
    with (IMG/'points.ply').open('w') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n')
        np.savetxt(f,points,fmt='%.4f')
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(10,4))
    a=fig.add_subplot(121,projection='3d'); a.scatter(*points.T,s=5,c=points[:,2],cmap='viridis'); a.set_title(f'Point cloud: {len(points)} XYZ points')
    b=fig.add_subplot(122,projection='3d'); b.add_collection3d(Poly3DCollection(vertices[faces],facecolors='#63b5ad',edgecolors='#17334a',alpha=.45)); b.set_title('Mesh: vertices + 12 triangle faces')
    for a in [a,b]:
        a.set(xlim=(-1.2,1.2),ylim=(-1.2,1.2),zlim=(-1.2,1.2),xlabel='X',ylabel='Y',zlabel='Z'); a.set_box_aspect((1,1,1))
    savefig('pointcloud-mesh.png')
    print('Assets generated:', IMG, MEDIA)

if __name__ == '__main__':
    main()
