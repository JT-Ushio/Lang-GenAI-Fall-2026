"""Offline numerical demonstrations: features, CTC, attention and edit distance."""
from itertools import product
from pathlib import Path
import wave
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def read_wav(path):
    with wave.open(str(path), 'rb') as f:
        assert f.getsampwidth() == 2 and f.getnchannels() == 1
        rate = f.getframerate()
        samples = np.frombuffer(f.readframes(f.getnframes()), dtype='<i2').astype(np.float32) / 32768
    return rate, torch.from_numpy(samples.copy())


def features(samples):
    """Same feature computation as Whisper for 16 kHz mono; 80-bin official filters."""
    window = torch.hann_window(400)
    stft = torch.stft(samples, n_fft=400, hop_length=160, window=window,
                      return_complex=True)
    power = stft[:, :-1].abs().square()
    filters = torch.from_numpy(np.load(ROOT / 'images/lecture_07/mel_filters.npz')['mel_80'])
    mel_power = filters @ power
    log_mel = mel_power.clamp_min(1e-10).log10()
    log_mel = torch.maximum(log_mel, log_mel.max() - 8)
    return power, (log_mel + 4) / 4


def collapse(path, blank=0):
    """Merge adjacent identical labels FIRST, then remove blank."""
    merged = [v for i, v in enumerate(path) if i == 0 or v != path[i - 1]]
    return tuple(v for v in merged if v != blank)


def enumerate_ctc(probs, target):
    total = 0.0
    valid = []
    for path in product(range(probs.shape[1]), repeat=probs.shape[0]):
        if collapse(path) == tuple(target):
            p = float(np.prod([probs[t, k] for t, k in enumerate(path)]))
            total += p
            valid.append((path, p))
    return total, valid


def ctc_example():
    # Three frames, vocabulary blank / 好. Greedy path is blank at all frames.
    probs = torch.tensor([[.6, .4], [.6, .4], [.6, .4]], dtype=torch.float64)
    probability, valid = enumerate_ctc(probs.numpy(), (1,))
    loss = F.ctc_loss(probs.log()[:, None, :], torch.tensor([1]),
                      torch.tensor([3]), torch.tensor([1]), reduction='sum')
    assert abs(float(loss.exp().reciprocal()) - probability) < 1e-12
    totals = {}
    for path in product(range(2), repeat=3):
        label = collapse(path)
        totals[label] = totals.get(label, 0) + float(np.prod([probs[t,k] for t,k in enumerate(path)]))
    return probs, probability, valid, totals


def optimize_ctc(steps=100):
    """Optimize frame logits for ONE fixed target 看一看. No acoustic encoder is trained."""
    rng = torch.Generator().manual_seed(7)
    logits = torch.nn.Parameter(torch.randn(8, 1, 3, generator=rng) * .1)
    optimizer = torch.optim.Adam([logits], lr=.15)
    target = torch.tensor([1, 2, 1])
    history = []
    for step in range(steps + 1):
        log_probs = logits.log_softmax(-1)
        loss = F.ctc_loss(log_probs, target, torch.tensor([8]), torch.tensor([3]), reduction='sum')
        if step in (0, 10, 30, steps):
            history.append({'step': step, 'loss': float(loss.detach())})
        if step < steps:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    probs = logits.detach().softmax(-1)[:, 0]
    path = probs.argmax(-1).tolist()
    return probs, path, collapse(path), history


def attention_demo():
    # Deliberately designed scores to illustrate soft alignment, not learned evidence.
    scores = torch.tensor([[4., 3., 0., -1., -2., -2.],
                           [-2., -1., 3., 4., 0., -2.],
                           [-2., -2., -1., 0., 3., 4.]])
    weights = scores.softmax(-1)
    values = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 24
    return weights, values, weights @ values


def edit_distance(reference, hypothesis):
    dp = np.zeros((len(reference)+1, len(hypothesis)+1), dtype=int)
    dp[:, 0] = np.arange(len(reference)+1)
    dp[0] = np.arange(len(hypothesis)+1)
    for i, a in enumerate(reference, 1):
        for j, b in enumerate(hypothesis, 1):
            dp[i,j] = min(dp[i-1,j]+1, dp[i,j-1]+1, dp[i-1,j-1]+(a != b))
    return int(dp[-1,-1]), dp


def shape_demo():
    """Randomly initialized Conv1d front-end: shape verification only."""
    with torch.random.fork_rng():
        torch.manual_seed(7)
        conv1 = torch.nn.Conv1d(80, 384, 3, padding=1)
        conv2 = torch.nn.Conv1d(384, 384, 3, stride=2, padding=1)
        with torch.no_grad():
            mel = torch.zeros(1, 80, 3000)
            h1 = F.gelu(conv1(mel))
            h2 = F.gelu(conv2(h1))
            sequence = h2.transpose(1, 2)
    return tuple(mel.shape), tuple(h1.shape), tuple(h2.shape), tuple(sequence.shape)
