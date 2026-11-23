"""CPU teaching models for duration, waveform reconstruction and music tokens."""
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from demos.lecture_07_speech import read_wav

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16000
NOTE_NAMES = ['宫', '商', '角', '徵', '羽']
MIDI_NOTES = [60, 62, 64, 67, 69]


def multitrack_demo():
    """Three original analytic stems; no source separation or pretrained inference."""
    seconds = 4
    time = torch.arange(seconds * SAMPLE_RATE) / SAMPLE_RATE
    beat_phase = time.remainder(.5)
    bass = .12 * torch.sin(2 * math.pi * 130.8128 * time)
    bass *= torch.exp(-5 * beat_phase)
    kick_phase = 2 * math.pi * (45 * beat_phase + 55 * .025 *
                                (1 - torch.exp(-beat_phase / .025)))
    drums = .18 * torch.sin(kick_phase) * torch.exp(-25 * beat_phase)
    notes = [60, 64, 67, 69, 67, 64, 62, 60]
    midi = torch.tensor(notes)[(time / .5).long().clamp_max(7)]
    frequency = 440 * 2 ** ((midi.float() - 69) / 12)
    phase = 2 * math.pi * torch.cumsum(frequency, 0) / SAMPLE_RATE
    lead = .12 * torch.sin(phase) * torch.sin(math.pi * beat_phase / .5).square()
    stems = torch.stack([bass, drums, lead])
    gains = torch.tensor([1., 1., 1.])
    mixture = (gains[:, None] * stems).sum(0)
    no_drums = (torch.tensor([1., 0., 1.])[:, None] * stems).sum(0)
    assert stems.shape == (3, 64000) and mixture.abs().max() < .95
    return stems, gains, mixture, no_drums


def length_regulator(features, durations):
    """Repeat each phoneme feature according to its number of acoustic frames."""
    durations = torch.as_tensor(durations, dtype=torch.long)
    assert features.shape[0] == durations.numel()
    assert bool((durations >= 0).all())
    return torch.repeat_interleave(features, durations, dim=0)


def voiced_tone(f0, duration=.6, harmonics=(1., .5, .25), gain=.25):
    """Analytic voiced vowel-like tone, with a continuous phase trajectory."""
    count = round(duration * SAMPLE_RATE)
    frequency = np.broadcast_to(np.asarray(f0, dtype=float), (count,))
    phase = 2 * np.pi * np.cumsum(frequency) / SAMPLE_RATE
    signal = np.zeros(count)
    for index, amplitude in enumerate(harmonics, 1):
        signal += amplitude * np.sin(index * phase)
    envelope = np.sin(np.linspace(0, np.pi, count)) ** .6
    return (gain * signal / sum(harmonics) * envelope).astype(np.float32)


def stft(waveform):
    return torch.stft(waveform, n_fft=400, hop_length=160,
                      window=torch.hann_window(400), return_complex=True)


def inverse_stft(spectrum, length):
    return torch.istft(spectrum, n_fft=400, hop_length=160,
                       window=torch.hann_window(400), length=length)


def griffin_lim(magnitude, length, iterations=32):
    """Estimate phase while holding a target magnitude spectrum fixed."""
    generator = torch.Generator().manual_seed(8)
    angle = torch.rand(magnitude.shape, generator=generator) * (2 * math.pi)
    spectrum = magnitude * torch.exp(1j * angle)
    for _ in range(iterations):
        waveform = inverse_stft(spectrum, length)
        estimated = stft(waveform)
        phase = estimated / estimated.abs().clamp_min(1e-8)
        spectrum = magnitude * phase
    return inverse_stft(spectrum, length)


def reconstruction_demo(waveform):
    spectrum = stft(waveform)
    exact = inverse_stft(spectrum, len(waveform))
    no_phase = inverse_stft(spectrum.abs().to(torch.complex64), len(waveform))
    with np.load(ROOT / 'images/lecture_07/mel_filters.npz') as bank:
        filters = torch.from_numpy(bank['mel_80'])
    mel_power = filters @ spectrum.abs().square()
    estimated_power = (torch.linalg.pinv(filters) @ mel_power).clamp_min(0)
    recovered = griffin_lim(estimated_power.sqrt(), len(waveform))
    error = float((exact - waveform).abs().max())
    assert error < 1e-5
    return {'exact': exact, 'zero_phase': no_phase, 'mel_recovered': recovered,
            'max_exact_error': error, 'mel_shape': list(mel_power.shape)}


def residual_quantize(vectors, codebooks):
    """Fixed-codebook RVQ demonstration. Codebooks have shape [Q, K, D]."""
    residual = vectors.clone()
    reconstructed = torch.zeros_like(vectors)
    codes, errors = [], []
    for book in codebooks:
        distance = (residual[:, None, :] - book[None, :, :]).square().sum(-1)
        ids = distance.argmin(-1)
        correction = book[ids]
        reconstructed += correction
        residual -= correction
        codes.append(ids)
        errors.append(float(residual.square().mean()))
    return torch.stack(codes, dim=0), reconstructed, errors


def rvq_demo():
    vectors = torch.tensor([[-.85, .3], [.35, -.6], [.8, .85]])
    axis = torch.tensor([-.75, -.25, .25, .75])
    coarse = torch.cartesian_prod(axis, axis)
    fine_axis = torch.tensor([-.125, 0., .125])
    fine = torch.cartesian_prod(fine_axis, fine_axis)
    # Equal-size codebooks simplify the batched shape. Coarse book includes zeros.
    fine = torch.cat([fine, torch.zeros(7, 2)])
    return vectors, torch.stack([coarse, fine])


def delay_pattern(codes):
    """Delay q-th codebook by q positions, padding empty cells with -1."""
    books, frames = codes.shape
    delayed = torch.full((books, frames + books - 1), -1, dtype=codes.dtype)
    for q in range(books):
        delayed[q, q:q + frames] = codes[q]
    return delayed


class TinyMelodyLM(nn.Module):
    """Two-layer causal Transformer over five symbolic notes and BOS/EOS."""
    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(7, 32)
        self.position = nn.Embedding(10, 32)
        self.condition = nn.Embedding(2, 32)
        layer = nn.TransformerEncoderLayer(32, 4, 64, dropout=0., batch_first=True)
        self.backbone = nn.TransformerEncoder(layer, 2)
        self.output = nn.Linear(32, 7)

    def forward(self, tokens, condition):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token(tokens) + self.position(positions)
        hidden = hidden + self.condition(condition)[:, None, :]
        mask = torch.triu(torch.ones(tokens.shape[1], tokens.shape[1],
                                     dtype=torch.bool, device=tokens.device), diagonal=1)
        return self.output(self.backbone(hidden, mask=mask))


def melody_dataset():
    """Ten fixed, cyclic five-note phrases, labelled by ascent or descent."""
    rows, conditions = [], []
    for direction in (0, 1):
        for start in range(5):
            step = 1 if direction == 0 else -1
            phrase = [(start + step * i) % 5 for i in range(5)]
            rows.append([5] + phrase + [6])  # 5=BOS, 6=EOS
            conditions.append(direction)
    return torch.tensor(rows), torch.tensor(conditions)


@torch.no_grad()
def generate_melody(model, condition, first_note):
    tokens = torch.tensor([[5, first_note]])
    for _ in range(7):
        logits = model(tokens, torch.tensor([condition]))[:, -1]
        next_id = int(logits.argmax(-1))
        if next_id == 6:
            break
        tokens = torch.cat([tokens, torch.tensor([[next_id]])], dim=1)
    return tokens[0, 1:].tolist()


def train_melody_model(steps=240):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(8)
            model = TinyMelodyLM()
            sequences, condition = melody_dataset()
            inputs, labels = sequences[:, :-1], sequences[:, 1:]
            labels = labels.clone()
            labels[:, 0] = -100  # The first note is supplied as part of the prompt.
            optimizer = torch.optim.AdamW(model.parameters(), lr=.004, weight_decay=.01)
            history = []
            for step in range(steps + 1):
                logits = model(inputs, condition)
                loss = F.cross_entropy(logits.reshape(-1, 7), labels.reshape(-1))
                if step in (0, 40, 120, steps):
                    history.append({'step': step, 'loss': float(loss.detach())})
                if step < steps:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
            model.eval()
            phrases = [generate_melody(model, 0, 0), generate_melody(model, 1, 4)]
            assert all(len(p) == 5 and all(0 <= n < 5 for n in p) for p in phrases)
            return model, {'history': history, 'phrases': phrases,
                           'parameters': sum(p.numel() for p in model.parameters()),
                           'training_phrases': 10, 'steps': steps}
    finally:
        torch.set_num_threads(previous_threads)


def render_melody(note_ids, bpm=100, harmonics=(1., .25, .1)):
    """Deterministically render symbolic note IDs using an additive synthesizer."""
    audio = []
    for token in note_ids:
        midi = MIDI_NOTES[token]
        frequency = 440 * 2 ** ((midi - 69) / 12)
        audio.append(voiced_tone(frequency, 60 / bpm, harmonics))
    return np.concatenate(audio)


def flow_matching_example():
    noise = torch.tensor([-1., .3, .8])
    target = torch.tensor([.2, -.5, .6])
    time = .4
    state = (1 - time) * noise + time * target
    velocity = target - noise
    final = noise.clone()
    for _ in range(10):
        final = final + .1 * velocity
    assert torch.allclose(final, target, atol=1e-6)
    return noise, target, state, velocity, final
