"""Original, deterministic musical examples for Lecture 08. No model downloads."""
import math
import numpy as np
import torch

SAMPLE_RATE = 16000


def midi_to_hz(notes):
    notes = torch.as_tensor(notes, dtype=torch.float32)
    return 440 * torch.pow(2., (notes - 69) / 12)


def render_score(notes, beats, bpm=120, harmonics=(1., .25, .1)):
    """One monophonic event at a time. Durations are beats, not seconds."""
    assert len(notes) == len(beats) and bpm > 0
    waveform = []
    for note, duration in zip(notes, beats):
        assert duration > 0
        count = round(float(duration) * 60 / bpm * SAMPLE_RATE)
        time = torch.arange(count) / SAMPLE_RATE
        frequency = float(midi_to_hz(note))
        tone = sum(amplitude * torch.sin(2 * math.pi * frequency * k * time)
                   for k, amplitude in enumerate(harmonics, 1))
        envelope = torch.sin(torch.linspace(0, math.pi, count)).square()
        waveform.append(.22 * tone / sum(harmonics) * envelope)
    return torch.cat(waveform)


def beat_clicks(beats_per_bar=4, bars=2, bpm=120):
    """Analytic click: the first beat of each bar is accented."""
    beat_samples = round(60 / bpm * SAMPLE_RATE)
    result = torch.zeros(beats_per_bar * bars * beat_samples)
    click_len = round(.045 * SAMPLE_RATE)
    time = torch.arange(click_len) / SAMPLE_RATE
    for beat in range(beats_per_bar * bars):
        accent = beat % beats_per_bar == 0
        click = (.35 if accent else .18) * torch.sin(2 * math.pi *
                  (1500 if accent else 1000) * time) * torch.exp(-90 * time)
        offset = beat * beat_samples
        result[offset:offset + click_len] += click
    return result


def render_chord(notes, seconds=2):
    time = torch.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    frequencies = midi_to_hz(notes)
    voices = torch.sin(2 * math.pi * frequencies[:, None] * time)
    envelope = torch.sin(torch.linspace(0, math.pi, time.numel())).square()
    return .24 * voices.mean(0) * envelope


def musical_examples():
    notes = [60, 62, 64, 67, 69, 67, 64, 60]
    straight = [1.] * 8
    varied = [.5, .5, 1., 2., .5, .5, 1., 2.]
    assert sum(straight) == sum(varied) == 8
    return {
        'music-meter-4': beat_clicks(4),
        'music-meter-3': beat_clicks(3),
        'music-rhythm-even': render_score(notes, straight),
        'music-rhythm-varied': render_score(notes, varied),
        'music-transposed': render_score([n + 2 for n in notes], straight),
        'music-major': render_chord([60, 64, 67]),
        'music-minor': render_chord([60, 63, 67]),
        'music-timbre-soft': render_score(notes, straight, harmonics=(1., .1, .03)),
        'music-timbre-bright': render_score(notes, straight, harmonics=(1., .65, .4, .25)),
    }


def music_features():
    notes = torch.tensor([60, 62, 64, 67, 69, 67, 64, 60])
    beats = torch.tensor([.5, .5, 1., 2., .5, .5, 1., 2.])
    starts = torch.cat([torch.zeros(1), beats.cumsum(0)[:-1]])
    shifted = notes + 2
    intervals = notes[1:] - notes[:-1]
    assert torch.equal(intervals, shifted[1:] - shifted[:-1])
    return notes, beats, starts, shifted, intervals
