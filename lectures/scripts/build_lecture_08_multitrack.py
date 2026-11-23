"""Rebuild the original multi-stem classroom example (no pretrained models)."""
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
from demos.lecture_08_audio import multitrack_demo, SAMPLE_RATE


def main():
    font = Path('/System/Library/Fonts/PingFang.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size': 13, 'figure.facecolor': '#f8fafc',
                         'axes.facecolor': '#f8fafc', 'text.color': '#153553'})
    stems, gains, mix, muted = multitrack_demo()
    names = ['stem-bass', 'stem-drums', 'stem-melody', 'stem-mix', 'stem-no-drums']
    for name, waveform in zip(names, [*stems, mix, muted]):
        with wave.open(str(ROOT / 'videos/lecture_08' / (name + '.wav')), 'wb') as stream:
            stream.setparams((1, 2, SAMPLE_RATE, 0, 'NONE', 'not compressed'))
            stream.writeframes(np.rint(waveform.numpy() * 32767).astype('<i2').tobytes())
    fig, axes = plt.subplots(5, 1, figsize=(10, 7), sharex=True, sharey=True)
    labels = ['低音轨 · 原创合成', '鼓点轨 · 原创合成', '旋律轨 · 原创合成',
              '三轨相加 · 相同采样率与时间轴', '仅将鼓点增益设为零']
    time = np.arange(mix.numel()) / SAMPLE_RATE
    for ax, label, waveform in zip(axes, labels, [*stems, mix, muted]):
        ax.plot(time[::16], waveform.numpy()[::16], color='#167d8d', lw=.7)
        ax.set_title(label, loc='left')
        ax.set_ylim(-.4, .4)
        ax.set_ylabel('振幅')
    axes[-1].set_xlabel('时间 / 秒')
    fig.tight_layout()
    fig.savefig(ROOT / 'images/lecture_08/multitrack.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
