"""Regenerate the classroom Mel comparisons from the included PCM WAV clips."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demos.lecture_07_speech import mel_power_spectrogram, read_wav

AUDIO = ROOT / 'videos/lecture_07/mel-examples'
IMAGES = ROOT / 'images/lecture_07/mel-examples'


def draw_group(clips, output_name):
    spectra = []
    for key, title in clips:
        rate, waveform = read_wav(AUDIO / f'{key}.wav')
        assert rate == 16000
        _, mel_power = mel_power_spectrogram(waveform)
        duration = len(waveform) / rate
        spectra.append((title, duration, mel_power))

    peak = max(float(power.max()) for _, _, power in spectra)
    end = max(duration for _, duration, _ in spectra)
    rows = len(clips) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(10, 2.85 * rows), sharex=True,
                             sharey=True, layout='constrained')
    for index, (ax, (title, duration, power)) in enumerate(zip(axes.flat, spectra)):
        db = 10 * (power / peak).clamp_min(1e-7).log10()
        # Centered STFT frames are sampled every 10 ms, starting at t=0.
        hop_seconds = .01
        frame_end = (db.shape[1] - .5) * hop_seconds
        mesh = ax.imshow(db.numpy(), aspect='auto', origin='lower',
                         extent=(-.005, frame_end, .5, 80.5),
                         interpolation='nearest', vmin=-65, vmax=0, cmap='magma')
        # Crop the image to the actual audio interval on the shared time axis.
        mesh.set_clip_path(plt.Rectangle((0, .5), duration, 80, transform=ax.transData))
        ax.set_facecolor('#eef0f2')
        ax.set_xlim(0, end)
        ax.set_title(f'{title} · {duration:.2f} 秒', loc='left', fontweight='bold')
        if index % 2 == 0:
            ax.set_ylabel('梅尔频带编号')
        if index // 2 == rows - 1:
            ax.set_xlabel('时间 / 秒')
    fig.colorbar(mesh, ax=axes, label='相对组内峰值 / dB', shrink=.95)
    fig.savefig(IMAGES / output_name, dpi=170)
    plt.close(fig)


def main():
    font = Path('/System/Library/Fonts/PingFang.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'DejaVu Sans']
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                         'text.color': '#153553', 'axes.labelcolor': '#153553'})
    IMAGES.mkdir(parents=True, exist_ok=True)
    draw_group([('bird', '鸟叫'), ('rain', '轻雨')], 'nature.png')
    draw_group([('timmie-zh', '中文'), ('timmie-ja', '日语'),
                ('timmie-en', '英语'), ('timmie-ko', '韩语')], 'timmie.png')


if __name__ == '__main__':
    main()
