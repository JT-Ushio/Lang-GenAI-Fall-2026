"""Build original music-focused listening examples and scientific figures."""
from pathlib import Path
import sys
import wave
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demos.lecture_08_music import musical_examples, music_features, midi_to_hz, SAMPLE_RATE


def save(name):
    plt.tight_layout()
    plt.savefig(ROOT/'images/lecture_08'/name, dpi=160, bbox_inches='tight')
    plt.close()


def main():
    font = Path('/System/Library/Fonts/PingFang.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size': 13, 'figure.facecolor': '#f8fafc',
                         'axes.facecolor': '#f8fafc', 'text.color': '#153553'})
    for name, samples in musical_examples().items():
        assert samples.isfinite().all() and samples.abs().max() < .95
        with wave.open(str(ROOT/'videos/lecture_08'/f'{name}.wav'), 'wb') as stream:
            stream.setparams((1,2,SAMPLE_RATE,0,'NONE','not compressed'))
            stream.writeframes(np.rint(samples.numpy()*32767).astype('<i2').tobytes())
    fig, axes = plt.subplots(2,1,figsize=(9,4.5))
    for ax, meter in zip(axes,[4,3]):
        beats=np.arange(meter*2);ax.bar(beats, [1 if i%meter==0 else .5 for i in beats],
                                      color=['#167d8d' if i%meter==0 else '#bacfd6' for i in beats])
        ax.set_xticks(beats, [str(i%meter+1) for i in beats]);ax.set_ylim(0,1.2)
        ax.set_title(f'{meter}/4 拍 · 每小节 {meter} 个四分音符拍',loc='left');ax.set_ylabel('重音示意')
    axes[-1].set_xlabel('拍号中的计数');save('music-meter.png')
    notes, beats, starts, shifted, intervals = music_features()
    fig, axes=plt.subplots(2,1,figsize=(9,4.8),sharex=True,sharey=True)
    for ax, ds in zip(axes,[np.ones(8),beats.numpy()]):
        ss=np.r_[0,np.cumsum(ds)[:-1]]
        for x,n,d in zip(ss,notes,ds):ax.broken_barh([(x,float(d)*.95)],(int(n)-.35,.7),facecolors='#167d8d')
        ax.set_ylabel('MIDI 音高');ax.set_yticks([60,64,67,69]);ax.grid(alpha=.2)
    axes[0].set_title('音高顺序相同：上图等时值，下图改变节奏',loc='left')
    axes[-1].set_xlabel('拍数');save('music-rhythm.png')
    fig,ax=plt.subplots(figsize=(9,4))
    ax.plot(notes,'o-',label='原旋律');ax.plot(shifted,'s--',label='整体升高 2 个半音')
    ax.set_xticks(range(8));ax.set_xlabel('音符位置');ax.set_ylabel('MIDI 音高');ax.legend()
    ax.set_title('移调保留相邻音程与节奏');save('music-transpose.png')
    fig,ax=plt.subplots(figsize=(8,4.5))
    for x,ns,label in [(0,[60,64,67],'C 大三和弦'),(1,[60,63,67],'C 小三和弦')]:
        ax.scatter([x]*3,ns,s=180,color='#167d8d' if x==0 else '#b06b42')
        for n in ns:ax.text(x+.08,n,f'{n} · {float(midi_to_hz(n)):.1f} Hz',va='center')
    ax.set_xlim(-.3,1.9);ax.set_xticks([0,1],['C–E–G','C–Eb–G（降 E）']);ax.set_ylabel('MIDI 音高')
    ax.set_title('同根音与五度：只改变中间的三度音');save('music-chords.png')
    fig,ax=plt.subplots(figsize=(10,3.8));colors=['#b7dce3','#167d8d','#b7dce3','#167d8d','#e3c3a8','#167d8d']
    segments=['A 主歌','B 副歌','A′ 主歌变体','B 副歌','C 桥段','B′ 副歌收束']
    for i,(label,c) in enumerate(zip(segments,colors)):
        ax.broken_barh([(i*4,3.9)],(.2,.6),facecolors=c);ax.text(i*4+1.95,.5,label,ha='center',va='center',fontsize=11)
    ax.set_xlim(0,24);ax.set_ylim(0,1);ax.set_yticks([]);ax.set_xlabel('示意小节位置（每段四小节）')
    ax.set_title('曲式：重复、变化、对比与回归');save('music-form.png')


if __name__=='__main__':
    main()
