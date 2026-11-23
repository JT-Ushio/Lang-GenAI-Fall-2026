"""Build numerical plots and offline audio. Run from the repository root."""
from pathlib import Path
import json
import sys
import wave

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demos.lecture_08_audio import (
    SAMPLE_RATE, NOTE_NAMES, MIDI_NOTES, read_wav, voiced_tone,
    reconstruction_demo, stft, length_regulator, rvq_demo, residual_quantize,
    delay_pattern, train_melody_model, render_melody, flow_matching_example,
)

IMAGE = ROOT / 'images/lecture_08'
AUDIO = ROOT / 'videos/lecture_08'
IMAGE.mkdir(parents=True, exist_ok=True)
AUDIO.mkdir(parents=True, exist_ok=True)


def write_wav(name, samples):
    samples = np.asarray(samples, dtype=np.float32)
    assert np.isfinite(samples).all()
    if np.max(np.abs(samples)) > .95:
        samples = samples * (.95 / np.max(np.abs(samples)))
    with wave.open(str(AUDIO / name), 'wb') as stream:
        stream.setparams((1, 2, SAMPLE_RATE, 0, 'NONE', 'not compressed'))
        stream.writeframes(np.rint(samples * 32767).astype('<i2').tobytes())


def save_figure(name):
    plt.tight_layout()
    plt.savefig(IMAGE / name, dpi=160, bbox_inches='tight', pad_inches=.2)
    plt.close()


def main():
    font = Path('/System/Library/Fonts/PingFang.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams.update({'font.size': 13, 'figure.facecolor': '#f8fafc',
                         'axes.facecolor': '#f8fafc', 'text.color': '#153553',
                         'axes.spines.top': False, 'axes.spines.right': False})

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for ax, name, label in zip(axes, ['speech-normal', 'speech-pause', 'speech-slow'],
                              ['自然读法', '加入标点停顿', '0.75 倍慢速']):
        rate, samples = read_wav(AUDIO / f'{name}.wav')
        assert rate == SAMPLE_RATE
        ax.plot(np.arange(len(samples)) / rate, samples, color='#167d8d', linewidth=.6)
        ax.set_title(f'{label} · {len(samples) / rate:.2f} 秒', loc='left')
        ax.set_ylim(-.6, .6)
        ax.set_ylabel('振幅')
    axes[-1].set_xlabel('时间 / 秒')
    save_figure('prosody.png')

    tone_specs = [('tone-low', 140, (1., .5, .25)),
                  ('tone-high', 210, (1., .5, .25)),
                  ('tone-bright', 140, (1., .9, .8, .6))]
    fig, axes = plt.subplots(3, 1, figsize=(9, 5.5), sharex=True)
    for ax, (name, f0, harmonics) in zip(axes, tone_specs):
        tone = voiced_tone(f0, harmonics=harmonics)
        write_wav(name + '.wav', tone)
        block = tone[1600:3200]
        power = abs(np.fft.rfft(block * np.hanning(len(block)))) ** 2
        frequency = np.fft.rfftfreq(len(block), 1 / SAMPLE_RATE)
        ax.plot(frequency, power, color='#167d8d')
        ax.set_xlim(0, 1500)
        ax.set_ylabel('功率')
        ax.set_title(f'基频 {f0} Hz，谐波比例 {harmonics}', loc='left')
    axes[-1].set_xlabel('频率 / Hz')
    save_figure('pitch-timbre.png')

    duration = .7
    timeline = np.linspace(0, duration, round(duration * SAMPLE_RATE))
    contours = [np.full(len(timeline), 170.),
                np.linspace(100, 200, len(timeline)),
                np.interp(timeline, [0, .35, .7], [160, 90, 160]),
                np.linspace(200, 90, len(timeline))]
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (label, contour) in enumerate(zip(['平调', '升调', '先降后升', '降调'], contours), 1):
        write_wav(f'contour-{i}.wav', voiced_tone(contour, duration))
        ax.plot(timeline, contour, label=label)
    ax.set_xlabel('时间 / 秒')
    ax.set_ylabel('程序设定的基频 / Hz')
    ax.legend()
    save_figure('tone-contours.png')

    units = torch.eye(4)
    expanded = length_regulator(units, [2, 3, 1, 2])
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.imshow(expanded.T, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    ax.set_yticks(range(4), ['我们', '明天', '这里', '见面'])
    ax.set_xlabel('展开后的时间位置')
    ax.set_title('每个位置保留对应单位的特征')
    save_figure('duration-regulator.png')

    _, speech = read_wav(AUDIO / 'speech-normal.wav')
    reconstruction = reconstruction_demo(speech)
    write_wav('speech-zero-phase.wav', reconstruction['zero_phase'].numpy())
    write_wav('speech-mel-griffin.wav', reconstruction['mel_recovered'].numpy())
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.2), sharex=True)
    for ax, signal, label in zip(axes,
                                [speech, reconstruction['zero_phase'], reconstruction['mel_recovered']],
                                ['原始语音', '用零相位重建', '梅尔伪逆与 Griffin–Lim']):
        power = stft(signal).abs().square().clamp_min(1e-8)
        ax.imshow(10 * power.log10(), origin='lower', aspect='auto', cmap='magma',
                  extent=[0, len(signal) / SAMPLE_RATE, 0, 8000], vmin=-60, vmax=20)
        ax.set_title(label, loc='left')
        ax.set_ylabel('频率 / Hz')
    axes[-1].set_xlabel('时间 / 秒')
    save_figure('reconstruction.png')

    vectors, books = rvq_demo()
    codes, reconstructed, errors = residual_quantize(vectors, books)
    coarse = books[0][codes[0]]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(books[0, :, 0], books[0, :, 1], color='#bdcbd6', label='一级码本')
    ax.scatter(vectors[:, 0], vectors[:, 1], marker='*', s=130, color='#d16b36', label='目标向量')
    ax.scatter(coarse[:, 0], coarse[:, 1], marker='s', s=75, color='#167d8d', label='一级重建')
    ax.scatter(reconstructed[:, 0], reconstructed[:, 1], marker='x', s=95, color='#5d58ac', label='两级重建')
    for a, b in zip(coarse, reconstructed):
        ax.plot([a[0], b[0]], [a[1], b[1]], color='#5d58ac')
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_xlabel('第一维'); ax.set_ylabel('第二维'); ax.legend()
    ax.set_title(f'固定码本，MSE {errors[0]:.4f} 降至 {errors[1]:.4f}')
    save_figure('rvq.png')

    codes = torch.arange(24).reshape(4, 6)
    delayed = delay_pattern(codes)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.imshow(delayed, aspect='auto', cmap='Blues', vmin=-1, vmax=23)
    for (i, j), value in np.ndenumerate(delayed.numpy()):
        ax.text(j, i, '空' if value < 0 else str(value), ha='center', va='center',
                color='white' if value > 13 else '#153553')
    ax.set_yticks(range(4), ['码本 1', '码本 2', '码本 3', '码本 4'])
    ax.set_xlabel('延迟排列位置')
    save_figure('delay-pattern.png')

    motif = [0, 2, 1, 4, 3, 2, 0, 1]
    write_wav('motif-80.wav', render_melody(motif, bpm=80))
    write_wav('motif-120.wav', render_melody(motif, bpm=120))
    write_wav('motif-bright.wav', render_melody(motif, bpm=80, harmonics=(1., .9, .7)))
    edited = motif.copy()
    edited[2] = 2
    write_wav('motif-edited.wav', render_melody(edited, bpm=80))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for bpm, offset, color in [(80, .08, '#167d8d'), (120, -.08, '#d16b36')]:
        for i, token in enumerate(motif):
            start = i * 60 / bpm
            ax.plot([start, start + 60 / bpm * .9], [MIDI_NOTES[token] + offset] * 2,
                    linewidth=5, color=color, label=f'{bpm} BPM' if i == 0 else None)
    ax.set_xlabel('时间 / 秒'); ax.set_ylabel('MIDI 音高编号'); ax.legend()
    save_figure('tempo.png')

    _, report = train_melody_model()
    for name, phrase in zip(['melody-forward.wav', 'melody-backward.wav'], report['phrases']):
        write_wav(name, render_melody(phrase))
    fig, ax = plt.subplots(figsize=(9, 3.7))
    ax.plot([r['step'] for r in report['history']], [r['loss'] for r in report['history']], 'o-', color='#167d8d')
    ax.set_xlabel('参数更新次数'); ax.set_ylabel('监督区交叉熵')
    ax.set_title('给定首音，学习后续音符与结束标记')
    save_figure('tiny-training.png')
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.3), sharex=True, sharey=True)
    for ax, phrase, name in zip(axes, report['phrases'], ['从宫前进', '从羽后退']):
        ax.step(np.arange(6), [MIDI_NOTES[n] for n in phrase] + [MIDI_NOTES[phrase[-1]]],
                where='post', color='#167d8d', linewidth=3)
        ax.set_yticks(MIDI_NOTES, NOTE_NAMES)
        ax.set_ylabel(name)
    axes[-1].set_xlabel('音符位置')
    save_figure('tiny-pianoroll.png')

    noise, target, _, velocity, _ = flow_matching_example()
    fig, ax = plt.subplots(figsize=(7, 4))
    for t in np.linspace(0, 1, 6):
        point = (1 - t) * noise + t * target
        ax.scatter(point[0], point[1], color=plt.cm.viridis(t), s=70)
        ax.text(point[0] + .02, point[1] + .02, f't={t:.1f}')
    ax.plot([noise[0], target[0]], [noise[1], target[1]], color='#aab8c4', zorder=0)
    ax.set_xlabel('潜变量第一维'); ax.set_ylabel('潜变量第二维')
    ax.set_title('给定噪声与目标的直线训练路径')
    save_figure('flow-path.png')

    report['exact_stft_error'] = reconstruction['max_exact_error']
    report['speech_mel_shape'] = reconstruction['mel_shape']
    report['rvq_errors'] = errors
    (ROOT / 'demos/lecture_08_result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
